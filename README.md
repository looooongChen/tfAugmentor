
# tfAugmentor
An image augmentation library for tensorflow. The libray is designed to be easily used with tf.data.Dataset. The augmentor accepts tf.data.Dataset object or a nested tuple of numpy array. 

## Augmentations
| **Original** | **Flip** | **Rotation** | **Translation** |
|:---------|:---------|:---------| :-------- |
| ![original](/demo/image/plant_grid.png) | ![demo_flip](/demo/demo_flip.png) | ![demo_rotation](/demo/demo_rotation.png) | ![demo_translation](/demo/demo_translation.png) |
| **Crop** | **Elactic Deform** |  |  |
| ![demo_crop](/demo/demo_crop.png) | ![demo_elastic](/demo/demo_elastic.png) |  |  |
| **Gaussian Blur**  | **Contrast** | **Gamma** | 
| ![demo_blur](/demo/demo_blur.png) | ![demo_contrast](/demo/demo_contrast.png) | ![demo_gamma](/demo/demo_gamma.png) |  |

| **Random Rotation** | **Random Translation** | **Random Crop** |
|:---------|:---------|:---------| 
| ![demo_ratation](/demo/gif/demo_random_rotation.gif) | ![demo_translation](/demo/gif/demo_random_translation.gif) | ![demo_crop](/demo/gif/demo_random_crop.gif) |
| **Random Contrast** | **Random Gamma** | **Elastic Deform** |
| ![demo_contrast](/demo/gif/demo_random_contrast.gif) | ![demo_gamma](/demo/gif/demo_random_gamma.gif) | ![demo_elastic](/demo/gif/demo_random_elastic.gif) |



## Deforming augmentations
| Homographic transform | Euclid transform |
|:---------|:--------------------|
| ![da_homographic](/samples/doc/da_homographic.gif) | ![da_euclid](/samples/doc/da_euclid.gif) |
| Elastic deformation | Random distortion |
| ![da_elastic](/samples/doc/da_elastic.gif) | ![da_distortion](/samples/doc/da_distortion.gif) |

## Installation
tfAugmentor is written in Python and can be easily installed via:
```python
pip install tfAugmentor
```
To run tfAugmentor properly, the following library should be installed:
- tensorflow (developed under tf 2.4), should work with 2.x version, 1.x version is not supported
- numpy (currently numpy=1.20 leads error of tf.meshgrid, please use another version)

## Quick Start
tfAugmentor is implemented to work seamlessly with tf.data. The tf.data.Dataset object can be directly processed by tfAugmentor. But you can also use it independently as a off-line augmentation tool.

To instantiate an `Augmentor` object, three arguments are required:

```python
class Augmentor(object):
    def __init__(self, signature, image=[], label_map=[]):
		...
```

- signature: a nested tuple of string, representing the structure of the dataset to be processesd e.g. ('image', 'segmentation') or keys of a dictionary, if your dataset is in a python dictionary form
- image, label: only the items in these two lists will be augmented, segmentation masks should be put in the label list so that the labels will kept valid

### simple example
```python
import tfAugmentor as tfaug

# new tfAugmentor object
aug = tfaug.Augmentor(signature=('image', 'semantic_mask'), image=['image'], label=['semantic_mask'])

# add augumentation operations
aug.flip_left_right(probability=0.5)
aug.rotate90(probability=0.5)
aug.elastic_deform(strength=2, scale=20, probability=1)

# assume we have two numpy arrays
X_image = ... # shape [batch, height, width, channel]
Y_semantic_mask = ... # shape [batch, height, width, 1]

# create tf.data.Dataset object
tf_dataset = tf.data.Dataset.from_tensor_slices((X_image, Y_semantic_mask)))
# do the actual augmentation
ds1 = aug(tf_dataset)

# or you can directly pass the numpy arrays, a tf.data.Dataset object will be returned 
ds2 = aug((X_image, Y_semantic_mask)), keep_size=True)
```

If you pass the data as a python dictionary, the signature should be the list/tuple of keys. For example:

```python
import tfAugmentor as tfaug

# new tfAugmentor object
aug = tfaug.Augmentor(signature=('image', 'semantic_mask'), image=['image'], label=['semantic_mask'])

# add augumentation operations
aug.flip_left_right(probability=0.5)
aug.rotate90(probability=0.5)
aug.elastic_deform(strength=2, scale=20, probability=1)

# assume we have three numpy arrays
X_image = ... # shape [batch, height, width, channel]
Y_semantic_mask = ... # shape [batch, height, width, 1]

ds_dict = {'image': X_image,
           'semantic_mask': Y_semantic_mask}
# create tf.data.Dataset object
tf_dataset = tf.data.Dataset.from_tensor_slices(ds_dict)
# do the actual augmentation
ds1 = aug(tf_dataset)

# or directly pass the data
ds2 = aug(ds_dict)
```


Note:
- All added augmentations will be executed one by one, but you can create multiply tfAugmentor to relize different augmentations in parallel
- itmes not in the 'image' or 'label' list will be untouched

### complex example

```python
import tfAugmentor as tfaug

# since 'class' is neither in 'image' nor in 'label', it will not be touched 
aug1 = tfaug.Augmentor((('image_rgb', 'image_depth'), ('semantic_mask', 'class')), image=['image_rgb', 'image_depth'], label=['semantic_mask'])
aug1 = tfaug.Augmentor((('image_rgb', 'image_depth'), ('semantic_mask', 'class')), image=['image_rgb', 'image_depth'], label=['semantic_mask'])

# add different augumentation operations to aug1 and aug2 
aug1.flip_left_right(probability=0.5)
aug1.random_crop_resize(sacle_range=(0.7, 0.9), probability=0.5)
aug2.elastic_deform(strength=2, scale=20, probability=1)

# assume we have the 1000 data samples
X_rgb = ...  # shape [1000 x 512 x 512 x 3]
X_depth = ... # shape [1000 x 512 x 512 x 1]
Y_semantic_mask = ... # shape [1000 x 512 x 512 x 1]
Y_class = ... # shape [1000 x 1]

# create tf.data.Dataset object
ds_origin = tf.data.Dataset.from_tensor_slices(((X_rgb, X_depth), (Y_semantic_mask, Y_class))))
# do the actual augmentation
ds1 = aug1(ds_origin)
ds2 = aug2(ds_origin)
# combine them
ds = ds_origin.concatenate(ds1)
ds = ds.concatenate(ds1)

```

## Main Features

### Mirroring
```python
aug.flip_left_right(probability=1) # flip the image left right  
aug.flip_up_down(probability=1) # flip the image up down
```
### Rotation
```python
a.rotate90(probability=1) # rotate by 90 degree clockwise
a.rotate180(probability=1) # rotate by 180 degree clockwise
a.rotate270(probability=1) # rotate by 270 degree clockwise
a.rotate(angle, probability=1) # rotate by *angel* degree, angle: scale in degree
a.random_rotate(probability=1) # randomly rotate the image
```

### Translation
```python
a.translate(offset, probability=1): # offset: [x, y]
a.random_translate(translation_range=[-100, 100], probability=1):
```

### Crop and Resize
```python
a.random_crop(scale_range=([0.5, 0.8], preserve_aspect_ratio=False, probability=1) # randomly crop a sub-image and resize to the original image size
```

### Elastic Deformation
```python
a.elastic_deform(scale=10, strength=200, probability=1)
```

### Photometric Adjustment
```python
a.random_contrast(contrast_range=[0.6, 1.4], probability=1)
a.random_gamma(gamma_range=[0.5, 1.5], probability=1)
```

### Noise
```python
a.gaussian_blur(sigma=2, probability=1)
```


## Caution
- If .batch() of tf.data.Dataset is used before augmentation, please set drop_remainder=True. Oherwise, the batch_size will be set to None. The augmention of tfAgmentor requires the batch_size dimension    