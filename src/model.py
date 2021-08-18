import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
from torch.nn.init import xavier_normal

from src.CrossmodalTransformer import MULTModel
from src.StoG import *
from src.GraphCAGE import *


class GCN_CAPS_Model(nn.Module):
    def __init__(self, args, label_dim, t_in, a_in, v_in, T_t, T_a, T_v,
                 MULT_d,
                 vertex_num,
                 dim_capsule,
                 routing,
                 dropout):
        super(GCN_CAPS_Model, self).__init__()
        self.d_c = dim_capsule
        self.n = vertex_num
        self.T_t = T_t
        self.T_a = T_a
        self.T_v = T_v

        # encode part
        self.CrossmodalTransformer = MULTModel(args, t_in, a_in, v_in, MULT_d, dropout)
        # transformation from sequence to graph
        self.StoG = CapsuleSequenceToGraph(args, MULT_d, dim_capsule, vertex_num, routing, T_t, T_a, T_v)
        # Graph aggregate
        self.GraphAggregate = GraphCAGE(args, MULT_d, dim_capsule, vertex_num, routing, T_t, T_a, T_v)
        # decode part
        self.fc1 = nn.Linear(in_features=3*dim_capsule*2, out_features=2*dim_capsule)
        self.fc2 = nn.Linear(in_features=2*dim_capsule, out_features=label_dim)

    def forward(self, text, audio, video, batch_size):
        Z_T, Z_A, Z_V = self.CrossmodalTransformer(text, audio, video)
        text_vertex, audio_vertex, video_vertex, adj_t, adj_a, adj_v = self.StoG(Z_T, Z_A, Z_V, batch_size)
        logits = self.GraphAggregate(text_vertex, audio_vertex, video_vertex, adj_t, adj_a, adj_v, batch_size)
        output1 = torch.tanh(self.fc1(logits))
        preds = self.fc2(output1) * 10
        return preds
