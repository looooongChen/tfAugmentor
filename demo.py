from skimage.io import imread, imsave
import tensorflow as tf
import numpy as np
import tfAugmentor as ta

img = imread('./demo/cell.png')
img = np.expand_dims(np.expand_dims(img, 2), 0)
mask = imread('./demo/mask.png')
mask = np.expand_dims(np.expand_dims(mask, 2), 0).astype(np.uint8)
print(img.shape, mask.shape)

img_tf = tf.placeholder(tf.uint8, shape=(1, None, None, 1))
mask_tf = tf.placeholder(tf.uint8, shape=(1, None, None, 1))

tensor_list = {'img': img_tf,
                'gt': mask_tf}
aug = ta.Augmentor(tensor_list, label=['gt'])
aug.elastic_deform(1, 3, 10)
aug2 = ta.Augmentor(tensor_list, label=['gt'])
aug2.rotate180(1)
aug3 = ta.Augmentor(tensor_list, label=['gt'])
out = aug.merge([aug2, aug3])

for i in range(5):
    with tf.Session() as sess:
        o = sess.run(out, feed_dict={img_tf: img, mask_tf: mask})

    print(o['img'].shape, o['gt'].shape)

    im = o['img']
    imsave('./demo/res_'+str(i)+'_img1.png', np.squeeze(im[0,:,:,:]))
    imsave('./demo/res_'+str(i)+'_img2.png', np.squeeze(im[1,:,:,:]))
    imsave('./demo/res_'+str(i)+'_img3.png', np.squeeze(im[2,:,:,:]))
    # imsave('./res_'+str(i)+'_gt.png', np.squeeze(o['gt']))