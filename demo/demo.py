from skimage.io import imread, imsave
import tensorflow as tf
import numpy as np
import tfAugmentor as ta
# from Augmentor import Augmentor

# img = imread('./cell.png')
img = np.expand_dims(imread('./cell.png'), axis=-1)
imgs = np.repeat(np.expand_dims(img, axis=0), 11, axis=0) 
mask = np.expand_dims(imread('./mask.png'), axis=-1)
masks = np.repeat(np.expand_dims(mask, axis=0), 11, axis=0) 

aa = ta.Augmentor(('image', 'mask'), image=['image'], label=['mask'])
aa.flip_left_right(probability=0.5)
# aa.flip_up_down(probability=0.5)
# aa.rotate90(probability=1)
# aa.rotate180(probability=1)
# aa.rotate270(probability=1)
aa.elastic_deform(strength=2, scale=20, probability=0.5)
# aa.rotate(45, probability=1)
# aa.random_rotate(probability=1)
# aa.random_crop([0.5, 0.8], probability=1)

ds = tf.data.Dataset.from_tensor_slices((imgs, masks))
ds = aa(ds)
i = 0
for img, mask in ds:
    imsave('./res/img_'+str(i)+'.png', np.squeeze(img))
    imsave('./res/mask_'+str(i)+'.png', np.squeeze(mask))
    i += 1

# ds = aa((imgs, masks))
# for i in range(len(ds[0])):
#     imsave('./res/img_'+str(i)+'.png', np.squeeze(ds[0][i]))
#     imsave('./res/mask_'+str(i)+'.png', np.squeeze(ds[1][i]))