import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
from torch.nn.init import xavier_normal


class GraphCAGE(nn.Module):
    def __init__(self, args, MULT_d, dim_capsule, vertex_num, routing, T_t, T_a, T_v):
        super(GraphCAGE, self).__init__()
        self.n = vertex_num
        self.MULT_d = MULT_d
        self.d_c = dim_capsule
        self.routing = routing
        self.T_t = T_t
        self.T_a = T_a
        self.T_v = T_v
        # GCN part
        self.conv1d_fir = nn.Conv1d(self.d_c, self.d_c, 1)
        self.conv1d_sec = nn.Conv1d(self.d_c, self.d_c, 1)
        self.Wo_fir = nn.Parameter(torch.Tensor(self.d_c, self.d_c))
        self.Wo_sec = nn.Parameter(torch.Tensor(self.d_c, self.d_c))
        nn.init.xavier_normal(self.Wo_fir)
        nn.init.xavier_normal(self.Wo_sec)

        # capsule part
        self.W_fir_pc = nn.Parameter(torch.Tensor(self.n, self.d_c, self.d_c))
        self.W_sec_pc = nn.Parameter(torch.Tensor(self.n, self.d_c, self.d_c))
        nn.init.xavier_normal(self.W_fir_pc)
        nn.init.xavier_normal(self.W_sec_pc)

    def forward(self, text_vertex, audio_vertex, video_vertex, adj_t, adj_a, adj_v, batch_size):
        # the first round GCN
        text_fir = self.conv1d_fir(text_vertex.permute(0, 2, 1))
        audio_fir = self.conv1d_fir(audio_vertex.permute(0, 2, 1))
        video_fir = self.conv1d_fir(video_vertex.permute(0, 2, 1))
        text_fir = torch.bmm(adj_t, text_fir.permute(0, 2, 1))
        audio_fir = torch.bmm(adj_a, audio_fir.permute(0, 2, 1))
        video_fir = torch.bmm(adj_v, video_fir.permute(0, 2, 1))
        text_fir_out = torch.tanh(torch.matmul(text_fir, self.Wo_fir))
        audio_fir_out = torch.tanh(torch.matmul(audio_fir, self.Wo_fir))
        video_fir_out = torch.tanh(torch.matmul(video_fir, self.Wo_fir))

        # the second round GCN
        text_sec = self.conv1d_sec(text_fir_out.permute(0, 2, 1))
        audio_sec = self.conv1d_sec(audio_fir_out.permute(0, 2, 1))
        video_sec = self.conv1d_sec(video_fir_out.permute(0, 2, 1))
        text_sec = torch.bmm(adj_t, text_sec.permute(0, 2, 1))
        audio_sec = torch.bmm(adj_a, audio_sec.permute(0, 2, 1))
        video_sec = torch.bmm(adj_v, video_sec.permute(0, 2, 1))
        text_sec_out = torch.tanh(torch.matmul(text_sec, self.Wo_sec))
        audio_sec_out = torch.tanh(torch.matmul(audio_sec, self.Wo_sec))
        video_sec_out = torch.tanh(torch.matmul(video_sec, self.Wo_sec))

        # capsule part
        tpc_fir = (torch.bmm(text_fir_out.permute(1, 0, 2), self.W_fir_pc)).permute(1, 0, 2)
        apc_fir = (torch.bmm(audio_fir_out.permute(1, 0, 2), self.W_fir_pc)).permute(1, 0, 2)
        vpc_fir = (torch.bmm(video_fir_out.permute(1, 0, 2), self.W_fir_pc)).permute(1, 0, 2)
        tpc_sec = (torch.bmm(text_sec_out.permute(1, 0, 2), self.W_sec_pc)).permute(1, 0, 2)
        apc_sec = (torch.bmm(audio_sec_out.permute(1, 0, 2), self.W_sec_pc)).permute(1, 0, 2)
        vpc_sec = (torch.bmm(video_sec_out.permute(1, 0, 2), self.W_sec_pc)).permute(1, 0, 2)

        tpc_fir_temp = tpc_fir.detach()
        apc_fir_temp = apc_fir.detach()
        vpc_fir_temp = vpc_fir.detach()
        tpc_sec_temp = tpc_sec.detach()
        apc_sec_temp = apc_sec.detach()
        vpc_sec_temp = vpc_sec.detach()

        # routing coefficients
        b_t_fir = torch.zeros(batch_size, self.n, 1)
        b_a_fir = torch.zeros(batch_size, self.n, 1)
        b_v_fir = torch.zeros(batch_size, self.n, 1)
        b_t_sec = torch.zeros(batch_size, self.n, 1)
        b_a_sec = torch.zeros(batch_size, self.n, 1)
        b_v_sec = torch.zeros(batch_size, self.n, 1)

        # routing mechanism
        for r in range(self.routing+1):
            rc_t_fir = F.softmax(b_t_fir, 1)
            rc_a_fir = F.softmax(b_a_fir, 1)
            rc_v_fir = F.softmax(b_v_fir, 1)
            rc_t_sec = F.softmax(b_t_sec, 1)
            rc_a_sec = F.softmax(b_a_sec, 1)
            rc_v_sec = F.softmax(b_v_sec, 1)
            logits_t_fir = (tpc_fir_temp * rc_t_fir).sum(1).tanh().unsqueeze(1)
            logits_a_fir = (apc_fir_temp * rc_a_fir).sum(1).tanh().unsqueeze(1)
            logits_v_fir = (vpc_fir_temp * rc_v_fir).sum(1).tanh().unsqueeze(1)
            logits_t_sec = (tpc_sec_temp * rc_t_sec).sum(1).tanh().unsqueeze(1)
            logits_a_sec = (apc_sec_temp * rc_a_sec).sum(1).tanh().unsqueeze(1)
            logits_v_sec = (vpc_sec_temp * rc_v_sec).sum(1).tanh().unsqueeze(1)

            if r < self.routing:
                new = (tpc_fir_temp * logits_t_fir).sum(2).unsqueeze(-1)
                b_t_fir = b_t_fir + new

                new = (apc_fir_temp * logits_a_fir).sum(2).unsqueeze(-1)
                b_a_fir = b_a_fir + new

                new = (vpc_fir_temp * logits_v_fir).sum(2).unsqueeze(-1)
                b_v_fir = b_v_fir + new

                new = (tpc_sec_temp * logits_t_sec).sum(2).unsqueeze(-1)
                b_t_sec = b_t_sec + new

                new = (apc_sec_temp * logits_a_sec).sum(2).unsqueeze(-1)
                b_a_sec = b_a_sec + new

                new = (vpc_sec_temp * logits_v_sec).sum(2).unsqueeze(-1)
                b_v_sec = b_v_sec + new

        logits_t_fir = (tpc_fir * rc_t_fir).sum(1).tanh()
        logits_a_fir = (apc_fir * rc_a_fir).sum(1).tanh()
        logits_v_fir = (vpc_fir * rc_v_fir).sum(1).tanh()
        logits_t_sec = (tpc_sec * rc_t_sec).sum(1).tanh()
        logits_a_sec = (apc_sec * rc_a_sec).sum(1).tanh()
        logits_v_sec = (vpc_sec * rc_v_sec).sum(1).tanh()

        logits = torch.cat([logits_t_fir, logits_t_sec, logits_a_fir, logits_a_sec, logits_v_fir, logits_v_sec], -1)
        return logits
