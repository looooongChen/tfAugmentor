from skimage.io import imread, imsave
import tensorflow as tf
import numpy as np
# import tfAugmentor as ta
import tfAugmentor as tfaug

# img = imread('./cell.png')
img = np.expand_dims(imread('./image/cell.png'), axis=-1)
imgs = np.repeat(np.expand_dims(img, axis=0), 11, axis=0) 
mask = np.expand_dims(imread('./image/mask.png'), axis=-1)
masks1 = np.repeat(np.expand_dims(mask, axis=0), 11, axis=0)
masks2 = masks1.copy() 

# mask2 unchanged
aug = tfaug.Augmentor(('image', ('mask1', 'mask2')), image=['image'], label=['mask1'])
# aug.flip_left_right(probability=0.5)
# aug.flip_up_down(probability=0.5)
aug.rotate90(probability=1)
# aug.rotate90(probability=0.5)
# aug.rotate180(probability=1)
# aug.rotate270(probability=1)
# aug.rotate(70, probability=1)
# aug.random_rotate(probability=1)
# aug.random_crop([0.5, 0.8], probability=1)
aug.elastic_deform(strength=2, scale=20, probability=1)

# ds = aug((imgs, (masks1, masks2)), keep_size=False)
ds = tf.data.Dataset.from_tensor_slices((imgs, (masks1, masks2)))
ds = aug(ds, keep_size=False)
i = 0
for img, mask in ds:
    imsave('./img_'+str(i)+'.png', np.squeeze(img))
    imsave('./mask_'+str(i)+'.png', np.squeeze(mask[0]))
    imsave('./mask2_'+str(i)+'.png', np.squeeze(mask[1]))
    i += 1