from skimage.io import imread, imsave
import tensorflow as tf
import numpy as np
import tfAugmentor as ta

img = imread('./cell.png')
img = np.expand_dims(np.expand_dims(img, 2), 0)
mask = imread('./mask.png')
mask = np.expand_dims(np.expand_dims(mask, 2), 0).astype(np.uint8)
print(img.shape, mask.shape)

img_tf = tf.placeholder(tf.uint8, shape=(1, None, None, 1))
mask_tf = tf.placeholder(tf.uint8, shape=(1, None, None, 1))

tensor_list = {'img': img_tf,
                'gt': mask_tf}
aug = ta.Augmentor(tensor_list, label=['gt'])
out = aug.elastic_deform(1, 3, 10)

for i in range(10):
    with tf.Session() as sess:
        o = sess.run(out, feed_dict={img_tf: img, mask_tf: mask})

    print(o['img'].shape, o['gt'].shape)

    imsave('./res_'+str(i)+'_img.png', np.squeeze(o['img']))
    imsave('./res_'+str(i)+'_gt.png', np.squeeze(o['gt']))