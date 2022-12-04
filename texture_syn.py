import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import torch.optim as optim

import torchvision.models as models
import util
import argparse
from GAN_model.DiscriminatorP import D2 as PointNetPotential
from GAN_model.gradient_penalty import Grad_Penalty
import numpy as np
from sklearn import manifold
from sklearn.decomposition import PCA


args_ps = argparse.ArgumentParser(description='This code synthesis a part of textures in the input image')

## model
args_ps.add_argument('--style', default='texture/fibrous_0115.jpg', help='The input image')
args_ps.add_argument('--out_size', type=int, default=256)
args_ps.add_argument('--mode', default='full', choices=['full', 'part'], help='Select full for texture synthesis or part for partial texture synthesis.')
args_ps.add_argument('--num_steps', type=int, default=1500)
args_ps.add_argument('--style_weight', type=float, default=100000000)
args_ps.add_argument('--slice_weight', type=float, default=100000000)

args_ps.add_argument('--im_scale', type=int, default=1)
args_ps.add_argument('--Init_cor', type=str, default=None, help='Sample position. Texture near this position will get sampled.')

args_ps.add_argument('--patch_size', type=int, default=16, help='Sample size')
args_ps.add_argument('--lr_G', type=float, default=1e-3, help='Learning rate for image update.')

#### D training
args_ps.add_argument('--lr_D', type=float, default=1e-3, help='Learning rate for network update.')
args_ps.add_argument('--d_iter', type=int, default=30, help='Network update iteration in each step.')
args_ps.add_argument('--ratio', type=float, default=1., help='Upsample ratio of the input image.')


args = args_ps.parse_args()


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("training")

s_path = "Texture_Synthesis/{}_r_{}_cor{}_lrg_{}_slice{}".format(os.path.basename(args.style), args.ratio, args.Init_cor, args.lr_G, args.slice_weight)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

style_img = util.image_loader(args.style, args.im_scale)

if args.mode == 'part':
    neg = True
elif args.mode == 'full':
    neg = False
else:
    raise NotImplementedError

leaky = 0.2
p_mass = 1 / (style_img.shape[-1] * style_img.shape[-2])
point_mass = [p_mass * args.ratio, p_mass]
G_P = Grad_Penalty(100, point_mass, gamma=1, device='cuda')


util.imshow(style_img, title='Style Image', show=False)

logger.info('Processing image {} with shape {}'.format(args.style, style_img.shape))
logger.info('Generate image of shape {}'.format(args.out_size))


tsne = manifold.TSNE(n_components=2)
pca = PCA(n_components=2)

########### use pretrained weights
F = models.vgg19(pretrained=True).features.to(device).eval()
style_layers_default = ['conv_1', 'conv_2', 'pool_2', 'pool_4', 'pool_8']

# initialize the output image by noise or by padding the image patch at the given location.
if args.Init_cor is None:
    input_img = (torch.randn((1, 3, args.out_size, args.out_size)) * 0.1 + 0.5).clamp_(0.1, 0.9).to('cuda')
else:
    Patch_size = args.patch_size
    cor = [int(i) for i in args.Init_cor.split(',')]

    Patch = style_img[:,:,cor[0]: cor[0] + Patch_size, cor[1]:cor[1] + Patch_size]
    input_img = torch.tensor(np.tile(Patch.cpu().detach().clone().numpy(), (1, 1, args.out_size // Patch_size, args.out_size // Patch_size)))
    input_img += (torch.randn((1, 3, args.out_size, args.out_size)) * 0.05).clamp_(-0.05, 0.05)
    input_img = input_img.to('cuda').clamp_(0., 1.)

optimizer_G = optim.RMSprop([input_img], lr=args.lr_G)

D = PointNetPotential(leaky=leaky)
D = D.to('cuda')
D_Net_parames = [j for (i,j) in D.named_parameters() if i != 'h']
optimizerD = optim.RMSprop([
                        {'params': D_Net_parames},
                        # {'params': D.h, 'lr':args.lr_D * 10}
                        ], lr=args.lr_D)


def get_model(cnn, layer_name):
    
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    # normalization layer
    normalization = util.Normalization(normalization_mean, normalization_std).to(device)

    Record_feature = {}
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            layer = nn.AvgPool2d(kernel_size=2, stride=2)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        
        if name in layer_name:
            record_layer = util.DistributionR()
            model.add_module(name + '_record', record_layer)
            Record_feature.update({name:record_layer})

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], util.DistributionR):
            break

    model = model[:(i + 1)]

    return model, Record_feature


def run_style_transfer(cnn, optimizer, optimizerD, style_img, input_img, num_steps=1000,
                       style_weight=10000):

    print('Building the feature extraction model..')
    model, Record = get_model(cnn, style_layers_default )
    
    model(style_img)
    Style_dis = []
    for i in style_layers_default:
        Style_dis.append(Record[i].R.detach().clone().contiguous())

    print('Optimize Process ..')
    for run in tqdm(range(num_steps)):
        ### Train inner network D
        with torch.no_grad():
            model(input_img)

        Style_dis_syn = []
        for i in style_layers_default:
            Style_dis_syn.append(Record[i].R.contiguous())

        # pdb.set_trace()
        for i in Style_dis:
            i.requires_grad_(True)
        for i in Style_dis_syn:
            i.requires_grad_(True)

        for i in range(args.d_iter):
            potential_r = D(Style_dis, neg=neg)
            potential_f = D(Style_dis_syn, neg=neg)
            d_loss = util.cal_dloss(potential_r, potential_f, point_mass) / 100

            # logger.debug('Real potential: max {} min {}'.format(torch.max(potential_r).item(), torch.min(potential_r).item()))
            # logger.debug('Fake potential: max {} min {}'.format(torch.max(potential_f).item(), torch.min(potential_f).item()))

            if i == 0:
                gp_loss, M, g1, g2 = G_P(d_loss, Style_dis + Style_dis_syn)
            else:
                gp_loss = torch.tensor(0.)
                M = torch.tensor(0.)

            d_loss_all = d_loss + gp_loss
            logger.debug('Iter: ' + str(run) + ' d_iter ' + str(i) + ' d_loss: ' + str(np.array(d_loss.item()).round(6)) + '\t' +
                'gp_loss: ' + str(np.array(gp_loss.item()).round(6)) + '\t' +
                         ' M_grad:' + str(np.array(M.item()).round(6)))

            optimizerD.zero_grad()
            d_loss_all.backward()
            optimizerD.step()


        input_img.requires_grad_(True)
        model.requires_grad_(False)
        style_score = 0

        model(input_img)

        Style_dis_syn_G = []
        for i in style_layers_default:
            Style_dis_syn_G.append(Record[i].R.contiguous())

        potential_f = D(Style_dis_syn_G, neg=neg)
        style_score += -torch.sum(potential_f * point_mass[1]) / 100

        style_score *= style_weight

        loss = style_score

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            input_img.clamp_(0, 1)

        if run % 10 == 0:
            logger.info("run {}  Style Loss : {:4f} ".format(run, style_score.item()))
            util.imsave(input_img, s_path, run)
            util.imsave(g1.unsqueeze(0), s_path, str(run) + '_grad' )
            util.imsave(potential_r.squeeze(0), s_path, str(run) + '_potential' )

    return input_img

output = run_style_transfer(F, optimizer_G, optimizerD, style_img, input_img, num_steps=args.num_steps, style_weight=args.style_weight)
