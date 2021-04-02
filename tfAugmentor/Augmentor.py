import tensorflow as tf
from tfAugmentor.operations import *
import numpy as np


class Augmentor(object):

    def __init__(self, signature, image=[], label=[]):
        self.signature = signature 
        self.signature_flatten = tf.nest.flatten(self.signature)
        self.image = image
        self.label = label
        self.ops = []
        # self.image_size = image_size
    
    def __call__(self, dataset, keep_size=True):

        '''
        Args:
            dataset: a tf.data.Dataset object that matches the signature
            keep_size: keep image size or not
        '''
        def transform(*item):
            item_flatten = tf.nest.flatten(item)
            for op in self.ops:
                item_flatten = op.run(item_flatten)
            item_aug = tf.nest.pack_sequence_as(item, item_flatten)

            return item_aug

        if len(self.image + self.label) != 0:
            if not isinstance(dataset, tf.data.Dataset):
                dataset = tf.data.Dataset.from_tensor_slices(dataset)
            # return dataset.map(tf.autograph.experimental.do_not_convert(transform))
            return dataset.map(transform)
        else:
            return None
    
    def flip_left_right(self, probability=1):
        self.ops.append(LeftRightFlip(self.signature_flatten, self.image, self.label, probability))

    def flip_up_down(self, probability=1):
        self.ops.append(UpDownFlip(self.signature_flatten, self.image, self.label, probability))

    def rotate90(self, probability=1):
        self.ops.append(Rotate90(self.signature_flatten, self.image, self.label, probability))

    def rotate180(self, probability=1):
        self.ops.append(Rotate180(self.signature_flatten, self.image, self.label, probability))

    def rotate270(self, probability=1):
        self.ops.append(Rotate270(self.signature_flatten, self.image, self.label, probability))

    # def rotate(self, angle, probability=1):
    #     self.ops.append(Rotate(self.signature_flatten, self.image, self.label, angle, probability))

    # def random_rotate(self, probability=1):
    #     self.ops.append(RandomRotate(self.signature_flatten, self.image, self.label, probability))

    # def random_crop(self, scale_range, probability=1, preserve_aspect_ratio=False):
    #     r = SyncRandomCropRunner(probability, scale_range, preserve_aspect_ratio)
    #     for k in self.image:
    #         r.sync(k, RandomCrop('bilinear'))
    #     for k in self.label:
    #         r.sync(k, RandomCrop('nearest'))
    #     self.funcs.append(r)
    
    def gaussian_blur(self, sigma=2, probability=1):
        self.ops.append(GaussianBlur(self.signature_flatten, self.image, self.label, sigma, probability))

    # def elastic_deform(self, strength, scale, probability=1):
    #     r = SyncElasticDeformRunner(probability, strength, scale)
    #     for k in self.image:
    #         r.sync(k, ElasticDeform('bilinear'))
    #     for k in self.label:
    #         r.sync(k, ElasticDeform('nearest'))
    #     self.funcs.append(r)

    

if __name__ == "__main__":
    import numpy as np
    s = ('a', ('b', 'c'), ('d', ('e', 'f')))
    ds = (1, (2, 3), (4, (5,6)))
    ds_d = sig2dict(s, ds)
    print(ds_d)
    print(dict2sig(s, ds_d))
    for a in unzip_signature(s):
        print(a)

    from skimage.io import imread, imsave
    import numpy as np
    img = imread('./demo/cell.png')
    imgs = np.repeat(np.expand_dims(img, axis=0), 12, axis=0) 
    mask = imread('./demo/mask.png') 
    masks = np.repeat(np.expand_dims(mask, axis=0), 12, axis=0) 

    ds = tf.data.Dataset.from_tensor_slices((imgs, masks))
    aa = Augmentor(('image', 'mask'))
    aa.flip_left_right(probability=1)
    ds = aa(ds)
    i = 0
    for img, mask in ds:
        imsave('./img_'+str(i)+'.png', np.squeeze(img))
        imsave('./mask_'+str(i)+'.png', np.squeeze(mask))
        i += 1

    t = LeftRightFlip()

