import tensorflow as tf
from tfAugmentor.operations import *
import numpy as np


IMG = 0
LB = 1

class SignatureError(Exception):
    def __init__(self, message):
        super().__init__(message)

def unzip_signature(signature):
        if isinstance(signature, tuple) or isinstance(signature, list):
            for s in signature:
                yield from unzip_signature(s)
        else:
            yield signature

def sig2dict(signature, ds):
    
    def _unzip(signature, ds):
        sig_item = isinstance(signature, tuple) or isinstance(signature, list)
        ds_item = isinstance(ds, tuple) or isinstance(ds, list)
        if sig_item != ds_item:
            raise SignatureError("Data structure does not match the signature")
        if sig_item:
            for i in range(len(signature)):
                yield from _unzip(signature[i], ds[i])
        else:
            yield signature, ds
    
    ds_dict = {s: d for s, d in _unzip(signature, ds)}
    
    return ds_dict

def dict2sig(signature, ds_dict):
    ds = []
    for s in signature:
        if isinstance(s, tuple) or isinstance(s, list):
            ds.append(dict2sig(s, ds_dict))
        else:
            ds.append(ds_dict[s])
    return tuple(ds)


class Augmentor(object):

    def __init__(self, signature=None, image=[], label=[]):
        self.signature = signature
        self.image = image
        self.label = label
        self.funcs = []
        # self.image_size = image_size
    
    def __call__(self, dataset, keep_size=True):

        '''
        Args:
            nested stucture: should match the signature, image data of size B x H x W x C 
            keep_size: keep image size or not
        '''

        def transform(*ds):
            if isinstance(ds[0], dict):
                ds_dict = ds[0]
            else:
                ds_dict = sig2dict(self.signature, ds)
            sz = tf.shape(ds_dict[(self.image + self.label)[0]])

            for f in self.funcs:
                f.run(ds_dict)
            
            if keep_size:
                for k in self.image:
                    ds_dict[k] = reisz_image(ds_dict[k], sz[-3:-1], 'bilinear')
                for k in self.label:
                    ds_dict[k] = reisz_image(ds_dict[k], sz[-3:-1], 'nearest')

            if isinstance(ds[0], dict):
                return ds_dict
            else:
                return dict2sig(self.signature, ds_dict)

        if len(self.image + self.label) != 0:
            if not isinstance(dataset, tf.data.Dataset):
                dataset = tf.data.Dataset.from_tensor_slices(dataset)
            return dataset.map(transform)
        else:
            return None

    def flip_left_right(self, probability=1):
        r = Runner(probability)
        for k in self.label + self.image:
            r.sync(k, LeftRightFlip())
        self.funcs.append(r)

    def flip_up_down(self, probability=1):
        r = Runner(probability)
        for k in self.label + self.image:
            r.sync(k, UpDownFlip())
        self.funcs.append(r)  

    def rotate90(self, probability=1):
        r = Runner(probability)
        for k in self.label + self.image:
            r.sync(k, Rotate90())
        self.funcs.append(r) 

    def rotate180(self, probability=1):
        r = Runner(probability)
        for k in self.label + self.image:
            r.sync(k, Rotate180())
        self.funcs.append(r) 

    def rotate270(self, probability=1):
        r = Runner(probability)
        for k in self.label + self.image:
            r.sync(k, Rotate270())
        self.funcs.append(r) 

    def rotate(self, angle, probability=1):
        r = Runner(probability)
        for k in self.image:
            r.sync(k, Rotate(angle, 'bilinear'))
        for k in self.label:
            r.sync(k, Rotate(angle, 'nearest'))
        self.funcs.append(r)

    def random_rotate(self, probability=1):
        r = RandomRotateRunner(probability)
        for k in self.image:
            r.sync(k, RandomRotate('bilinear'))
        for k in self.label:
            r.sync(k, RandomRotate('nearest'))
        self.funcs.append(r)

    def random_crop(self, scale_range, probability=1):
        r = RandomCropRunner(probability, scale_range)
        for k in self.image:
            r.sync(k, RandomCrop('bilinear'))
        for k in self.label:
            r.sync(k, RandomCrop('nearest'))
        self.funcs.append(r)

    def elastic_deform(self, strength, scale, probability=1):
        r = ElasticDeformRunner(probability, strength, scale)
        for k in self.image:
            r.sync(k, ElasticDeform('bilinear'))
        for k in self.label:
            r.sync(k, ElasticDeform('nearest'))
        self.funcs.append(r)

    

if __name__ == "__main__":
    # import numpy as np
    # s = ('a', ('b', 'c'), ('d', ('e', 'f')))
    # ds = (1, (2, 3), (4, (5,6)))
    # ds_d = sig2dict(s, ds)
    # print(ds_d)
    # print(dict2sig(s, ds_d))
    # for a in unzip_signature(s):
    #     print(a)

    # from skimage.io import imread, imsave
    # import numpy as np
    # img = imread('./demo/cell.png')
    # imgs = np.repeat(np.expand_dims(img, axis=0), 12, axis=0) 
    # mask = imread('./demo/mask.png') 
    # masks = np.repeat(np.expand_dims(mask, axis=0), 12, axis=0) 

    # ds = tf.data.Dataset.from_tensor_slices((imgs, masks))
    # aa = Augmentor(('image', 'mask'))
    # aa.flip_left_right(probability=1)
    # ds = aa(ds)
    # i = 0
    # for img, mask in ds:
    #     imsave('./img_'+str(i)+'.png', np.squeeze(img))
    #     imsave('./mask_'+str(i)+'.png', np.squeeze(mask))
    #     i += 1

    t = LeftRightFlip()

