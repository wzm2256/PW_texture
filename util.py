
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


unloader = transforms.ToPILImage()  # reconvert into PIL image

def imshow(tensor, title=None, show=True):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    if show == True:
        plt.imshow(image)
        plt.show()


def imsave(tensor, path='', name='a'):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    if not os.path.isdir(path):
        os.makedirs(path)
    image.save(os.path.join(path, str(name)) + '.png')


def image_loader(image_name, scale, device='cuda'):
    image = Image.open(image_name).convert('RGB')
    im_shape = image.size
    loader = transforms.Compose([
        transforms.Resize((im_shape[1] // scale // 16 * 16, im_shape[0] // scale // 16 * 16)),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
        


class DistributionR(nn.Module):
    # This module records its input as its status.
    def __init__(self):
        super(DistributionR, self).__init__()

    def forward(self, input):
        self.R = input
        return input

def cal_dloss(potential1, potential2, point_mass):
    D_real_int = torch.sum(potential1 * point_mass[0])
    D_fake_int = torch.sum(potential2 * point_mass[1])
    d_loss = D_fake_int - D_real_int
    return d_loss

# def cal_var(X):
#     var_total = 0.
#     for i in X[:4]:
#         # var_total += torch.sum(torch.var(i, (-1, -2)))
#         var_total += torch.mean((i - torch.mean(i, (-1, -2), keepdim=True).detach()) ** 2, (-1, -2)).sum()
#         # pdb.set_trace()
#     return var_total


# def Extract(image, number):
#     size = image.shape
#     x_cor = torch.randint(0, size[2]-16, [number])
#     y_cor = torch.randint(0, size[3]-16, [number])
#
#     P = []
#     for i in range(number):
#         P.append(image[:, :, x_cor[i]: x_cor[i] + 16, y_cor[i]: y_cor[i] + 16])
#     out = torch.cat(P, 0)
#     return out

# class ContentLoss(nn.Module):
#
#     def __init__(self, target,):
#         super(ContentLoss, self).__init__()
#         self.target = target.detach()
#
#     def forward(self, input):
#         self.loss = F.mse_loss(input, self.target)
#         return input

# def gram_matrix(input):
#     a, b, c, d = input.size()  # a=batch size(=1)
#     # b=number of feature maps
#     # (c,d)=dimensions of a f. map (N=c*d)
#
#     features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
#
#     G = torch.mm(features, features.t())  # compute the gram product
#
#     return G.div(a * b * c * d)


# class StyleLoss(nn.Module):
#
#     def __init__(self, target_feature):
#         super(StyleLoss, self).__init__()
#         self.target = gram_matrix(target_feature).detach()
#
#     def forward(self, input):
#         G = gram_matrix(input)
#         self.loss = F.mse_loss(G, self.target)
#         return input

