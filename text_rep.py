from cgitb import small
import PIL.Image as Image
import numpy as np
import argparse
import pdb
import matplotlib.pyplot  as plt
import torchvision.transforms as transforms
import os

agsps = argparse.ArgumentParser()
agsps.add_argument('--image', default='mask_image/vincent.png')
agsps.add_argument('--mask', default='mask_image/1620_grad.png')
agsps.add_argument('--texture', default='mask_image/1.png')
agsps.add_argument('--name', default='a')

ags = agsps.parse_args()


unloader = transforms.ToPILImage()
I = Image.open(ags.image).convert('RGB')
I_ori = I.size
I = I.crop((0,0, min(I.size[0], 256), min(I.size[1], 256)))

M = Image.open(ags.mask).convert('L')
M = M.crop((0,0, min(M.size[0], 256), min(M.size[1], 256)))

# pdb.set_trace()
T = Image.open(ags.texture).convert('RGB')
T = T.crop((0,0, min(I_ori[0], 256), min(I_ori[1], 256)))

I = np.asarray(I, dtype=np.float) / 255
M = np.expand_dims(np.asarray(M, dtype=np.float) / 255, -1)
T = np.asarray(T, dtype=np.float) / 255

Small_ind = M < 0.2
M[Small_ind] = 0

new_I = I * (1-M) + M * T

plt.imsave('mask_image/' + ags.name + '.png', new_I)
plt.imsave('mask_image/' + ags.name + 'old.png', I)
plt.imsave('mask_image/' + ags.name + 'mask_old.png', np.squeeze(M, -1), cmap='gray' )

# image = Image.fromarray(new_I.astype(np.float32))
# image.save('new.png')

# image = Image.fromarray(I.astype(np.float32))
# image.save('old.png')


# plt.imshow(new_I)
# plt.show()

# plt.imshow(I)
# plt.show()