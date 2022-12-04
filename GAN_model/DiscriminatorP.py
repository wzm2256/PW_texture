import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# class MyRelu(nn.Module):
#     # use leaky relu or relu.
#     def __init__(self, leaky=None):
#         super(MyRelu, self).__init__()
#         self.relu = nn.LeakyReLU(negative_slope=leaky)
#
#     def forward(self, x):
#         return self.relu(x)


class D2(nn.Module):
    # a unet like concate between 64 dim layers
    # We wish to resize all input tensor to have the same spatial dimension, then concatenate their channel dimension.
    # But that will require too much GPU memory. Therefore, we first process each input tensor separately, and then
    # add them up before feed them to the second layer. This is equivalent to the concatenation approach.
    def __init__(self,  leaky=None):
        super(D2, self).__init__()

        Input_F = [64, 64, 64, 128, 256]
        Layers = [64, 32, 1]

        self.Input_P = nn.ModuleList([nn.Sequential() for _ in range(len(Input_F))])

        for i in range(len(self.Input_P)):
            self.Input_P[i] = nn.Sequential()
            self.Input_P[i].add_module('M1{}_conv_'.format(i), torch.nn.Conv2d(Input_F[i], Layers[0], 1))
            # self.Input_P[i].add_module('M1{}_leaky_'.format(i), MyRelu(leaky=leaky))

        self.Process = nn.Sequential()
        for i in range(len(Layers[:-2])):
            self.Process.add_module('P_conv_' + str(i), torch.nn.Conv2d(Layers[i], Layers[i+1], 1))
            self.Process.add_module('P_conv_leaky_' + str(i), nn.LeakyReLU(negative_slope=leaky))
        self.Process.add_module('P_conv_1', torch.nn.Conv2d(Layers[-2], Layers[-1], 1))

        self.leaky = leaky

    def forward(self, Dis_list, neg=True):
        Ori_size = Dis_list[0].shape[2]

        Second_list = []
        for Input_Layer, Input in zip(self.Input_P, Dis_list):
            Second_list.append(Input_Layer(Input))
        
        S_Second_List = []
        for i in Second_list:
            Scale = Ori_size // i.shape[2]
            S_Second_List.append(torch.repeat_interleave(torch.repeat_interleave(i, Scale, 2), Scale, 3))

        Feature = F.leaky_relu(torch.stack(S_Second_List, 0).sum(0), negative_slope=self.leaky, inplace=False)
        p_out = self.Process(Feature)
        
        if neg:
            out = -torch.abs(p_out)
        else:
            out = p_out

        return out


#
# class D1(nn.Module):
#     # a unet like concate between 64 dim layers
#     def __init__(self,  leaky=None, h=None,  N1=None, **kwargs):
#         super(D1, self).__init__()
#
#
#
#         Input_F = [64, 64, 128, 128, 256]
#         Layers = [64, 32, 1]
#
#         self.Input_P = nn.ModuleList([nn.Sequential() for _ in range(5)])
#
#         for i in range(len(self.Input_P)):
#             self.Input_P[i] = nn.Sequential()
#             self.Input_P[i].add_module('M1{}_conv_'.format(i), torch.nn.Conv2d(Input_F[i], Layers[0], 1))
#             # self.Input_P[i].add_module('M1{}_leaky_'.format(i), MyRelu(leaky=leaky))
#
#         self.Process = nn.Sequential()
#         for i in range(len(Layers[:-2])):
#             self.Process.add_module('P_conv_' + str(i), torch.nn.Conv2d(Layers[i], Layers[i+1], 1))
#             self.Process.add_module('P_conv_leaky_' + str(i), MyRelu(leaky=leaky))
#         self.Process.add_module('P_conv_1', torch.nn.Conv2d(Layers[-2], Layers[i-1], 1))
#
#         # self.h = Parameter(-torch.abs(torch.tensor(h, dtype=torch.float32)), requires_grad=True)
#         self.h = Parameter(torch.tensor(h, dtype=torch.float32), requires_grad=True)
#
#         self.leaky = leaky
#         # self.N1 = N1
#
#     def forward(self, Dis_list, clip=False, neg=True):
#         Ori_size = Dis_list[0].shape[2]
#
#         Second_list = []
#         for Input_Layer, Input in zip(self.Input_P, Dis_list):
#             Second_list.append(Input_Layer(Input))
#
#         S_Second_List = []
#         for i in Second_list:
#             Scale = Ori_size // i.shape[2]
#             S_Second_List.append(torch.repeat_interleave(torch.repeat_interleave(i, Scale, 2), Scale, 3))
#
#         Feature = F.leaky_relu(torch.stack(S_Second_List, 0).sum(0), negative_slope=self.leaky, inplace=False)
#         p_out = self.Process(Feature)
#
#         if neg:
#             out = -torch.abs(p_out)
#         else:
#             out = p_out
#
#         h = -torch.abs(self.h)
#         if clip == True:
#             P = torch.nn.functional.relu(out - h) + h
#         else:
#             P = out
#
#         return P

