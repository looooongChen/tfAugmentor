import tensorflow as tf
from skimage.io import imread, imsave
import math
from operations import gaussian, warp_image
import numpy as np


class tfAugmentor(object):

    def __init__(self, tensor_list, label=[]):
        self.tensors = {k: tensor for k, tensor in tensor_list.items()}
        self.types = {k: "image" for k, tensor in tensor_list.items()}
        self.out = dict(self.tensors)

        for l in label:
            self.types[l] = "label"

        self.tensor_shape = tf.shape(list(self.tensors.values())[0])

    def add_operation(self, funcs, prob):
        r = tf.random_uniform([], 0, 1)
        for k, tensor in self.out.items():
            self.out[k] = tf.cond(tf.greater(r, prob),
                                  lambda: tensor,
                                  lambda: funcs[k](tensor))
        return self.out

    def flip_left_right(self, prob):
        funcs = {}
        for k, img_type in self.types.items():
            funcs[k] = tf.image.flip_left_right
        return self.add_operation(funcs, prob)

    def flip_up_down(self, prob):
        funcs = {}
        for k, img_type in self.types.items():
            funcs[k] = tf.image.flip_up_down
        return self.add_operation(funcs, prob)

    def _rotate(self, prob, rotate_k):
        funcs = {}
        for k, img_type in self.types.items():
            funcs[k] = lambda input: tf.image.rot90(input, rotate_k)
        return self.add_operation(funcs, prob)

    def rotate90(self, prob):
        return self._rotate(prob, 1)

    def rotate180(self, prob):
        return self._rotate(prob, 2)

    def rotate270(self, prob):
        return self._rotate(prob, 3)

    def rotate(self, prob, angle):
        funcs = {}
        for k, img_type in self.types.items():
            if img_type == "label":
                funcs[k] = lambda input: tf.contrib.image.rotate(input, angle, interpolation='NEAREST')
            else:
                funcs[k] = lambda input: tf.contrib.image.rotate(input, angle, interpolation='BILINEAR')
        return self.add_operation(funcs, prob)

    def random_rotate(self, prob):
        funcs = {}
        angle = tf.random_uniform([self.tensor_shape[0]], 0, 2*math.pi)
        for k, img_type in self.types.items():
            if img_type == "label":
                funcs[k] = lambda input: tf.contrib.image.rotate(input, angle, interpolation='NEAREST')
            else:
                funcs[k] = lambda input: tf.contrib.image.rotate(input, angle, interpolation='BILINEAR')
        return self.add_operation(funcs, prob)

    def elastic_deform(self, prob, strength, scale):
        funcs = {}
        dx = tf.random_uniform([self.tensor_shape[0],
                                tf.floordiv(self.tensor_shape[1], scale),
                                tf.floordiv(self.tensor_shape[2], scale),
                                1], -1, 1)
        dy = tf.random_uniform([self.tensor_shape[0],
                                tf.floordiv(self.tensor_shape[1], scale),
                                tf.floordiv(self.tensor_shape[2], scale),
                                1], -1, 1)
        dx = gaussian(dx, 0, 5)
        dy = gaussian(dy, 0, 5)
        flow = strength*tf.concat([dx, dy], axis=-1)
        flow = tf.image.resize_bilinear(flow, self.tensor_shape[1:3])
        for k, img_type in self.types.items():
            if img_type == "label":
                funcs[k] = lambda input: warp_image(input, flow, interpolation='knearest')
            else:
                funcs[k] = lambda input: warp_image(input, flow, interpolation='bilinear')
        return self.add_operation(funcs, prob)

    def random_crop_resize(self, prob, scale_range=(0.5, 0.8)):
        funcs = {}
        size = tf.random_uniform([self.tensor_shape[0], 2], scale_range[0], scale_range[1])
        offset = tf.multiply(1-size, tf.random_uniform([self.tensor_shape[0], 2], 0, 1))
        boxes = tf.concat([offset, offset+size], axis=1)
        box_ind = tf.range(0, self.tensor_shape[0], delta=1, dtype=tf.int32)

        for k, img_type in self.types.items():
            if img_type == "label":
                funcs[k] = lambda input: tf.cast(
                    tf.image.crop_and_resize(input, boxes, box_ind,
                                             [self.tensor_shape[1], self.tensor_shape[2]],
                                             method='nearest'), input.dtype)
            else:
                funcs[k] = lambda input: tf.cast(
                    tf.image.crop_and_resize(input, boxes, box_ind,
                                             [self.tensor_shape[1], self.tensor_shape[2]],
                                             method='bilinear'), input.dtype)
        return self.add_operation(funcs, prob)

    def crop(self, prob, sz):
        sz = np.array(sz, np.int32)
        funcs = {}
        size = [tf.divide(sz[0], self.tensor_shape[1]), tf.divide(sz[1], self.tensor_shape[2])]
        size = tf.cast(tf.tile(tf.expand_dims(size, 0), [self.tensor_shape[0], 1]), tf.float32)
        offset = tf.multiply(1 - size, tf.random_uniform([self.tensor_shape[0], 2], 0, 1))
        boxes = tf.concat([offset, offset + size], axis=1)
        box_ind = tf.range(0, self.tensor_shape[0], delta=1, dtype=tf.int32)

        for k, img_type in self.types.items():
            if img_type == "label":
                funcs[k] = lambda input: tf.cast(
                    tf.image.crop_and_resize(input, boxes, box_ind, sz,
                                             method='nearest'), input.dtype)
            else:
                funcs[k] = lambda input: tf.cast(
                    tf.image.crop_and_resize(input, boxes, box_ind, sz,
                                             method='bilinear'), input.dtype)
        return self.add_operation(funcs, prob)


if __name__ == "__main__":
    import numpy as np
    img = imread('./test_images/img.png')
    img = np.expand_dims(np.expand_dims(img, 2), 0)
    mask = imread('./test_images/mask.png')
    mask = np.expand_dims(np.expand_dims(mask, 2), 0).astype(np.uint8)
    print(img.shape, mask.shape)

    img_tf = tf.placeholder(tf.uint8, shape=(1, None, None, 1))
    mask_tf = tf.placeholder(tf.uint8, shape=(1, None, None, 1))

    tensor_list = {'img': img_tf,
                   'gt': mask_tf}
    aug = tfAugmentor(tensor_list, label=['gt'])
    out = aug.crop(1, (600, 600))

    for i in range(10):
        with tf.Session() as sess:
            o = sess.run(out, feed_dict={img_tf: img, mask_tf: mask})

        print(o['img'].shape, o['gt'].shape)

        imsave('res_'+str(i)+'_img.png', np.squeeze(o['img']))
        imsave('res_'+str(i)+'_gt.png', np.squeeze(o['gt']))
