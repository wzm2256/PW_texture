import util
# import torch
import argparse
import os

args_ps = argparse.ArgumentParser()
args_ps.add_argument('--style', default='texture/fibrous_0115.jpg')
args_ps.add_argument('--im_scale', type=int, default=1)

args = args_ps.parse_args()



style_img = util.image_loader(args.style, args.im_scale, device='cpu')
print(style_img.shape)


util.imsave(style_img, path='Texture2', name=os.path.basename(args.style.split('.')[0]))

