# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Shijia Zhou (szho6430@uni.sydney.edu.au)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import torch.nn as nn
import torch



class JointsMSELossWithRegularization(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELossWithRegularization, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight
        self.sigmoid = nn.Sigmoid()
        self.m = nn.Softplus()

    def forward(self, output, target, target_weight,use_reg):
        # print(output.size)
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        heatmaps = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            heatmaps.append(heatmap_pred)
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        

        if use_reg is not None:
            if use_reg != False:
                a = use_reg
                a1 = use_reg
                
                raw_reg = self.criterion(heatmaps[0],heatmaps[1])

                # regularization = 0.00001*(1/raw_reg)/(raw_reg+1)
                # 对角函数
                a = torch.tensor(a).cuda()
                optimal_raw_reg = torch.tensor(0.0061)
                optimal_reg_size = torch.tensor(0.001)
                b = (1/(a*(optimal_raw_reg)**2.0)).cuda()
                unadjusted_reg_size = a*optimal_raw_reg + 1/(b*optimal_raw_reg)
                c = (torch.log(optimal_reg_size)/torch.log(unadjusted_reg_size)).cuda()


                regularization = ((a*raw_reg + 1/(b*raw_reg))**c) - (a*torch.sqrt(1/(a*b)) + 1/(b*torch.sqrt(1/(a*b))))**c
                loss += regularization

                a1 = torch.tensor(a1).cuda()
                b1 = (1/(a1*(optimal_raw_reg/2)**2.0)).cuda()
                unadjusted_reg_size = a1*(optimal_raw_reg/2) + 1/(b1*(optimal_raw_reg/2))
                c1 = (torch.log(optimal_reg_size)/torch.log(unadjusted_reg_size)).cuda()

                raw_reg1 = self.criterion(heatmaps[0],torch.zeros(heatmaps[0].shape).cuda())
                raw_reg += raw_reg1
                reg1 = ((a1*raw_reg1 + 1/(b1*raw_reg1))**c1) - (a1*torch.sqrt(1/(a1*b1)) + 1/(b1*torch.sqrt(1/(a1*b1))))**c1
                regularization += reg1
                loss += reg1

                raw_reg2 = self.criterion(heatmaps[1],torch.zeros(heatmaps[0].shape).cuda())
                raw_reg += raw_reg2
                reg2 = ((a1*raw_reg2 + 1/(b1*raw_reg2))**c1) - (a1*torch.sqrt(1/(a1*b1)) + 1/(b1*torch.sqrt(1/(a1*b1))))**c1
                regularization += reg2
                loss += reg2
                

            else:
                loss = loss
                raw_reg = self.criterion(heatmaps[0],heatmaps[1])
                regularization = torch.tensor(0)
                regularization = ((a*raw_reg + 1/(b*raw_reg))**c) - (a*torch.sqrt(1/(a*b)) + 1/(b*torch.sqrt(1/(a*b))))**c

                loss += regularization
        if use_reg is None:
            loss = loss
            raw_reg = self.criterion(heatmaps[0],heatmaps[1])
            regularization = torch.tensor(0)
        return loss / num_joints, regularization, raw_reg, heatmaps