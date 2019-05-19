import tensorflow as tf
import math
from .operations import gaussian, warp_image
import numpy as np


class Augmentor(object):

    def __init__(self, tensor_list, label=[]):
        self.tensors = {k: tensor for k, tensor in tensor_list.items()}
        self.types = {k: "image" for k, tensor in tensor_list.items()}
        self.out = dict(self.tensors)

        for l in label:
            self.types[l] = "label"

        self.tensor_shape = tf.shape(list(self.tensors.values())[0])

    def add_operation(self, funcs, probability):
        r = tf.random_uniform([], 0, 1)
        for k, tensor in self.out.items():
            self.out[k] = tf.cond(tf.greater(r, probability),
                                  lambda: tensor,
                                  lambda: funcs[k](tensor))
        return self.out

    def flip_left_right(self, probability):
        funcs = {}
        for k, _ in self.types.items():
            funcs[k] = tf.image.flip_left_right
        return self.add_operation(funcs, probability)

    def flip_up_down(self, probability):
        funcs = {}
        for k, _ in self.types.items():
            funcs[k] = tf.image.flip_up_down
        return self.add_operation(funcs, probability)

    def _rotate(self, probability, rotate_k):
        funcs = {}
        for k, _ in self.types.items():
            funcs[k] = lambda input: tf.image.rot90(input, rotate_k)
        return self.add_operation(funcs, probability)

    def rotate90(self, probability):
        return self._rotate(probability, 1)

    def rotate180(self, probability):
        return self._rotate(probability, 2)

    def rotate270(self, probability):
        return self._rotate(probability, 3)

    def rotate(self, probability, angle):
        funcs = {}
        for k, img_type in self.types.items():
            if img_type == "label":
                funcs[k] = lambda input: tf.contrib.image.rotate(input, angle, interpolation='NEAREST')
            else:
                funcs[k] = lambda input: tf.contrib.image.rotate(input, angle, interpolation='BILINEAR')
        return self.add_operation(funcs, probability)

    def random_rotate(self, probability):
        funcs = {}
        angle = tf.random_uniform([self.tensor_shape[0]], 0, 2*math.pi)
        for k, img_type in self.types.items():
            if img_type == "label":
                funcs[k] = lambda input: tf.contrib.image.rotate(input, angle, interpolation='NEAREST')
            else:
                funcs[k] = lambda input: tf.contrib.image.rotate(input, angle, interpolation='BILINEAR')
        return self.add_operation(funcs, probability)

    def elastic_deform(self, probability, strength, scale):
        strength = strength * 100
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
        return self.add_operation(funcs, probability)

    def random_crop_resize(self, probability, scale_range=(0.5, 0.8)):
        funcs = {}
        sz = tf.random_uniform([self.tensor_shape[0], 2], scale_range[0], scale_range[1])
        offset = tf.multiply(1-sz, tf.random_uniform([self.tensor_shape[0], 2], 0, 1))
        boxes = tf.concat([offset, offset+sz], axis=1)
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
        return self.add_operation(funcs, probability)

    def crop(self, probability, size):
        size = np.array(size, np.int32)
        funcs = {}
        sz = [tf.divide(size[0], self.tensor_shape[1]), tf.divide(size[1], self.tensor_shape[2])]
        sz = tf.cast(tf.tile(tf.expand_dims(sz, 0), [self.tensor_shape[0], 1]), tf.float32)
        offset = tf.multiply(1 - sz, tf.random_uniform([self.tensor_shape[0], 2], 0, 1))
        boxes = tf.concat([offset, offset + sz], axis=1)
        box_ind = tf.range(0, self.tensor_shape[0], delta=1, dtype=tf.int32)

        for k, img_type in self.types.items():
            if img_type == "label":
                funcs[k] = lambda input: tf.cast(
                    tf.image.crop_and_resize(input, boxes, box_ind, size,
                                             method='nearest'), input.dtype)
            else:
                funcs[k] = lambda input: tf.cast(
                    tf.image.crop_and_resize(input, boxes, box_ind, size,
                                             method='bilinear'), input.dtype)
        return self.add_operation(funcs, probability)

    def merge(self, augmentors):
        out = {}
        for k, tensor in self.out.items():
            tmp = []
            tmp.append(tensor)
            for a in augmentors:
                assert k in a.out
                tmp.append(a.out[k])
            out[k] = tf.concat(tmp, axis=0)
        return out