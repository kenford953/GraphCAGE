import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import numpy as np
from torch.nn.init import xavier_normal


class CapsuleSequenceToGraph(nn.Module):
    def __init__(self, args, MULT_d, dim_capsule, vertex_num, routing,
                 T_t, T_a, T_v):
        super(CapsuleSequenceToGraph, self).__init__()
        self.d_c = dim_capsule
        self.n = vertex_num
        self.routing = routing
        # self.pc_dropout = dropout
        # create primary capsule
        self.W_tpc = nn.Parameter(torch.Tensor(T_t, self.n, 2*MULT_d, self.d_c))
        self.W_apc = nn.Parameter(torch.Tensor(T_a, self.n, 2*MULT_d, self.d_c))
        self.W_vpc = nn.Parameter(torch.Tensor(T_v, self.n, 2*MULT_d, self.d_c))
        nn.init.xavier_normal(self.W_tpc)
        nn.init.xavier_normal(self.W_apc)
        nn.init.xavier_normal(self.W_vpc)

        # create adjacent matrix by self-attention
        self.WQt = nn.Parameter(torch.Tensor(self.d_c, self.d_c))
        self.WKt = nn.Parameter(torch.Tensor(self.d_c, self.d_c))
        self.WQa = nn.Parameter(torch.Tensor(self.d_c, self.d_c))
        self.WKa = nn.Parameter(torch.Tensor(self.d_c, self.d_c))
        self.WQv = nn.Parameter(torch.Tensor(self.d_c, self.d_c))
        self.WKv = nn.Parameter(torch.Tensor(self.d_c, self.d_c))
        nn.init.xavier_normal(self.WQt)
        nn.init.xavier_normal(self.WQa)
        nn.init.xavier_normal(self.WQv)
        nn.init.xavier_normal(self.WKt)
        nn.init.xavier_normal(self.WKa)
        nn.init.xavier_normal(self.WKv)

    def forward(self, text, audio, video, batch_size):
        # get dimensionality
        T_t = text.shape[0]
        T_a = audio.shape[0]
        T_v = video.shape[0]
        # create primary capsule
        text_pri_caps = (torch.einsum('tbj, tnjd->tbnd', text, self.W_tpc)).permute(1, 0, 2, 3)
        audio_pri_caps = (torch.einsum('tbj, tnjd->tbnd', audio, self.W_apc)).permute(1, 0, 2, 3)
        video_pri_caps = (torch.einsum('tbj, tnjd->tbnd', video, self.W_vpc)).permute(1, 0, 2, 3)

        # routing mechanism does not participate in back propagation
        text_pri_caps_temp = text_pri_caps.detach()
        audio_pri_caps_temp = audio_pri_caps.detach()
        video_pri_caps_temp = video_pri_caps.detach()

        # begin routing
        for r in range(self.routing+1):
            if r == 0:
                b_t = torch.zeros(batch_size, T_t, self.n)  # initialize routing coefficients
                b_a = torch.zeros(batch_size, T_a, self.n)
                b_v = torch.zeros(batch_size, T_v, self.n)
            rc_t = F.softmax(b_t, 2)
            rc_a = F.softmax(b_a, 2)
            rc_v = F.softmax(b_v, 2)

            text_vertex = torch.tanh(torch.sum(text_pri_caps_temp * rc_t.unsqueeze(-1), 1))
            audio_vertex = torch.tanh(torch.sum(audio_pri_caps_temp * rc_a.unsqueeze(-1), 1))
            video_vertex = torch.tanh(torch.sum(video_pri_caps_temp * rc_v.unsqueeze(-1), 1))

            # update routing coefficients
            if r < self.routing:
                last = b_t
                new = ((text_vertex.unsqueeze(1)) * text_pri_caps_temp).sum(3)
                b_t = last + new

                last = b_a
                new = (audio_vertex.unsqueeze(1) * audio_pri_caps_temp).sum(3)
                b_a = last + new

                last = b_v
                new = (video_vertex.unsqueeze(1) * video_pri_caps_temp).sum(3)
                b_v = last + new

        # create vertex using the routing coefficients in final round
        text_vertex = torch.tanh(torch.sum(text_pri_caps * rc_t.unsqueeze(-1), 1))
        audio_vertex = torch.tanh(torch.sum(audio_pri_caps * rc_a.unsqueeze(-1), 1))
        video_vertex = torch.tanh(torch.sum(video_pri_caps * rc_v.unsqueeze(-1), 1))

        # use self-attention to create adjacent matrix
        Q = torch.matmul(text_vertex, self.WQt)
        K = torch.matmul(text_vertex, self.WKt)
        adj_t = torch.eye(self.n) + F.relu(torch.bmm(Q, K.permute(0, 2, 1)) / self.d_c)
        Q = torch.matmul(audio_vertex, self.WQa)
        K = torch.matmul(audio_vertex, self.WKa)
        adj_a = torch.eye(self.n) + F.relu(torch.bmm(Q, K.permute(0, 2, 1)) / self.d_c)
        Q = torch.matmul(video_vertex, self.WQv)
        K = torch.matmul(video_vertex, self.WKv)
        adj_v = torch.eye(self.n) + F.relu(torch.bmm(Q, K.permute(0, 2, 1)) / self.d_c)
        return text_vertex, audio_vertex, video_vertex, adj_t, adj_a, adj_v


class AttentionSequenceToGraph(nn.Module):
    def __init__(self, args):
        super(AttentionSequenceToGraph, self).__init__()
        self.d = args.MULT_d
        self.WQ_t = nn.Parameter(torch.Tensor(args.MULT_d*2, args.MULT_d*2))
        self.WK_t = nn.Parameter(torch.Tensor(args.MULT_d*2, args.MULT_d*2))
        self.WQ_a = nn.Parameter(torch.Tensor(args.MULT_d*2, args.MULT_d*2))
        self.WK_a = nn.Parameter(torch.Tensor(args.MULT_d*2, args.MULT_d*2))
        self.WQ_v = nn.Parameter(torch.Tensor(args.MULT_d*2, args.MULT_d*2))
        self.WK_v = nn.Parameter(torch.Tensor(args.MULT_d*2, args.MULT_d*2))

        xavier_normal(self.WQ_t)
        xavier_normal(self.WQ_a)
        xavier_normal(self.WQ_v)
        xavier_normal(self.WK_t)
        xavier_normal(self.WK_a)
        xavier_normal(self.WK_v)

    def forward(self, text, audio, video, batch_size):
        np.set_printoptions(threshold=np.inf)
        T_t = text.shape[0]
        T_a = audio.shape[0]
        T_v = video.shape[0]
        text = text.permute(1, 0, 2)
        audio = audio.permute(1, 0, 2)
        video = video.permute(1, 0, 2)
        Q = torch.matmul(text, self.WQ_t)
        K = torch.matmul(text, self.WK_t)
        adj_t = torch.eye(T_t) + F.relu(torch.bmm(Q, K.permute(0, 2, 1)) / self.d)
        Q = torch.matmul(audio, self.WQ_a)
        K = torch.matmul(audio, self.WK_a)
        adj_a = torch.eye(T_a) + F.relu(torch.bmm(Q, K.permute(0, 2, 1)) / self.d)
        Q = torch.matmul(video, self.WQ_v)
        K = torch.matmul(video, self.WK_v)
        adj_v = torch.eye(T_v) + F.relu(torch.bmm(Q, K.permute(0, 2, 1)) / self.d)
        return text, audio, video, adj_t, adj_a, adj_v