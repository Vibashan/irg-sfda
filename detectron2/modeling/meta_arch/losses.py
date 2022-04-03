"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07*2):
        super(GraphConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        dim_in = 2048
        feat_dim = 2048
        self.head_1 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        self.head_2 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self, t_feat, s_feat, graph_cn, labels=None, mask=None):    

        qx = graph_cn.graph.wq(s_feat)
        kx = graph_cn.graph.wk(s_feat)        
        sim_mat = qx.matmul(kx.transpose(-1, -2))
        dot_mat = sim_mat.detach().clone()

        thresh = 0.5
        dot_mat -= dot_mat.min(1, keepdim=True)[0]
        dot_mat /= dot_mat.max(1, keepdim=True)[0]
        mask = ((dot_mat>thresh)*1).detach().clone()
        mask.fill_diagonal_(1)

        anchor_feat = self.head_1(s_feat)
        contrast_feat = self.head_2(s_feat)

        anchor_feat = F.normalize(anchor_feat, dim=1)
        contrast_feat = F.normalize(contrast_feat, dim=1)

        ss_anchor_dot_contrast = torch.div(torch.matmul(anchor_feat, contrast_feat.T), self.temperature)  ##### torch.Size([6, 6])
        logits_max, _ = torch.max(ss_anchor_dot_contrast, dim=1, keepdim=True)  ##### torch.Size([6, 1]) - contains max value along dim=1
        ss_graph_logits = ss_anchor_dot_contrast - logits_max.detach()

        ss_graph_all_logits = torch.exp(ss_graph_logits)
        ss_log_prob = ss_graph_logits - torch.log(ss_graph_all_logits.sum(1, keepdim=True))
        ss_mean_log_prob_pos = (mask * ss_log_prob).sum(1) / mask.sum(1)  
    
        # loss
        ss_loss = - (self.temperature / self.base_temperature) * ss_mean_log_prob_pos
        ss_loss = ss_loss.mean()

        return ss_loss




