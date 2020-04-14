from skimage.io import imread, imsave
import tensorflow as tf
import numpy as np
# import tfAugmentor as ta
import tfAugmentor as tfaug

# img = imread('./cell.png')
img = np.expand_dims(imread('./image/cell.png'), axis=-1)
imgs = np.repeat(np.expand_dims(img, axis=0), 11, axis=0) 
mask = np.expand_dims(imread('./image/mask.png'), axis=-1)
masks = np.repeat(np.expand_dims(mask, axis=0), 11, axis=0) 

aug = tfaug.Augmentor(('image', 'mask'), image=['image'], label=['mask'])
aug.flip_left_right(probability=0.5)
# aug.flip_up_down(probability=0.5)
aug.rotate90(probability=1)
# aug.rotate90(probability=0.5)
# aug.rotate180(probability=1)
# aug.rotate270(probability=1)
# aug.rotate(70, probability=1)
# aug.random_rotate(probability=1)
# aug.random_crop([0.5, 0.8], probability=1)
aug.elastic_deform(strength=2, scale=20, probability=1)

# ds = aug((imgs, masks), keep_size=True)
ds = tf.data.Dataset.from_tensor_slices((imgs, masks))
ds = aug(ds, keep_size=True)
i = 0
for img, mask in ds:
    imsave('./img_'+str(i)+'.png', np.squeeze(img))
    imsave('./mask_'+str(i)+'.png', np.squeeze(mask))
    i += 1