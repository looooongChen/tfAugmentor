
# tfAugmentor
An image augmentation library for tensorflow. The libray is designed to be easily used with tf.data.Dataset. The augmentor accepts tf.data.Dataset object or a nested tuple of numpy array. 

## new features
- support tf 2.x
- API change for easier use, can directly process tf.data.Dataset object

## Installation
tfAugmentor is written in Python and can be easily installed via:
```python
pip install tfAugmentor
```
To run tfAugmentor properly, the following library should be installed as well:
- tensorflow (developed under tf 2.1), should work with 2.x version, 1.x version is not supported
- tensorflow-addons
- numpy (developed under numpy 1.18)
- tensorflow-probability (optional)

## Quick Start
tfAugmentor is implemented to work seamlessly with tf.data. The tf.data.Dataset object can be directly processed by tfAugmentor. But you can also use it independently as a off-line augmentation tool.

To instantiate an `Augmentor` object, three arguments are required:

```python
class Augmentor(object):
    def __init__(self, signature, image=[], label_map=[]):
		...
```

- signature: a nested tuple of string, representing the structure of the dataset to be processesd e.g. ('image', 'segmentation')
- image, label: only the items in these two lists will be augmented, segmentation masks should be put in the label list so that the labels will kept valid

### simple example
```python
import tfAugmentor as tfaug

# new tfAugmentor object
aug = tfaug.Augmentor(('image', 'semantic_mask'), image=['image'], label=['semantic_mask'])

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
ds1 = aug(tf_dataset, keep_size=True)

# or you can directly pass the numpy arrays, a tf.data.Dataset object will be returned 
ds2 = aug((X_image, Y_semantic_mask)), keep_size=True)
```

If you pass the data as a python dictionary, the signature is not necessary any more. For example:

```python
import tfAugmentor as tfaug

# new tfAugmentor object
aug = tfaug.Augmentor(image=['image'], label=['semantic_mask'])

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
ds1 = aug(tf_dataset, keep_size=True)

# or directly pass the data
ds2 = aug(ds_dict, keep_size=True)
```


Note:
- All added augmentations will be executed one by one, but you can create multiply tfAugmentor to relize different augmentations in parallel
- all data should have a 4-D shape of `[batch, height, width, channels]` with the first dimension being the same, unprocessed items (itmes not in the 'image' or 'label' list) can have any dataset shape  

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
ds1 = aug1(ds_origin, keep_size=True)
ds2 = aug2(ds_origin, keep_size=True)
# combine them
ds = ds_origin.concatenate(ds1)
ds = ds.concatenate(ds1)

```

## Main Features

### Mirroring
```python
aug.flip_left_right(probability) # flip the image left right  
aug.flip_up_down(probability) # flip the image up down
```
### Rotating
```python
a.rotate90(probability) # rotate by 90 degree clockwise
a.rotate180(probability) # rotate by 180 degree clockwise
a.rotate270(probability) # rotate by 270 degree clockwise
a.rotate(angle, probability) # rotate by *angel* degree clockwise
a.random_rotate(probability) # randomly rotate the image
```
### crop and resize
```python
a.random_crop(scale_range=(0.5, 0.8), probability) # randomly crop a sub-image and resize to the same size of the original image
a.random_crop(scale_range=0.8, probability) # fixed crop size, random crop position
```

### elastic deformation
```
a.elastic_deform(strength, scale, probability)
```