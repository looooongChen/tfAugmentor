from skimage.io import imread, imsave
import tensorflow as tf
import numpy as np
import tfAugmentor as tfaug

# img = imread('./demo/image/cell.png')
img = imread('./demo/image/plant.png')
imgs = np.repeat(np.expand_dims(img, axis=0), 11, axis=0) 
# mask = np.expand_dims(imread('./demo/image/mask_cell.png'), axis=-1)
mask = np.expand_dims(imread('./demo/image/mask_plant.png'), axis=-1)
masks1 = np.repeat(np.expand_dims(mask, axis=0), 11, axis=0)
masks2 = masks1.copy() 

#### nested tuple input ####
aug = tfaug.Augmentor(('image', ('mask1', 'mask2')), image=['image'], label=['mask1'])
# aug2 = tfaug.Augmentor(('image', ('mask1', 'mask2')), image=['image'], label=['mask1'])

# aug.flip_left_right(probability=0.5)
# aug.flip_up_down(probability=0.5)
# aug.rotate90(probability=0.5)
# aug.rotate180(probability=0.5)
# aug.rotate270(probability=0.5)
aug.gaussian_blur(sigma=5)

# aug.rotate(70, probability=0.5)
# aug.random_rotate(probability=1)
# aug.random_crop([0.5, 0.8], probability=1, preserve_aspect_ratio=True)
# aug.elastic_deform(strength=3, scale=10, probability=1)


# ds = aug((imgs, (masks1, masks2)), keep_size=False)
ds = tf.data.Dataset.from_tensor_slices((imgs, (masks1, masks2)))
# ds = tf.data.Dataset.from_tensor_slices((imgs, masks2, masks1))
ds_aug = aug(ds, keep_size=False)
# ds_aug2 = aug2(ds, keep_size=False)

# ds = ds_aug.concatenate(ds_aug2)
ds = ds_aug

i = 0
for img, mask in ds:
    print(i)
    imsave('./demo/img_'+str(i)+'.png', np.squeeze(img))
    imsave('./demo/mask_'+str(i)+'.png', 20*np.squeeze(mask[0]))
    imsave('./demo/mask2_'+str(i)+'.png', 20*np.squeeze(mask[1]))
    i += 1

#### dictionary input ####

# aug = tfaug.Augmentor(image=['image'], label=['mask1'])
# aug.rotate90(probability=0.5)
# aug.elastic_deform(strength=2, scale=20, probability=1)

# ds_dict = {'image': imgs,
#            'mask1': masks1,
#            'mask2': masks2}
# ds = aug(ds_dict, keep_size=False)

# for i, item in enumerate(ds):
#     imsave('./img_'+str(i)+'.png', np.squeeze(item['image']))
#     imsave('./mask_'+str(i)+'.png', np.squeeze(item['mask1']))
#     imsave('./mask2_'+str(i)+'.png', np.squeeze(item['mask2']))