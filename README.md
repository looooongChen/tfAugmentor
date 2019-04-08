
still under construction, coming soon
# tfAugmentor
An image augmentation library for tensorflow. All operations are implemented as pure tensorflow graph operations. Thus, tfAugmentor can be easily combined with any tensorflow graph, such as tf.Data, for on-the-fly data augmentation. 
Of cousrse, you can also use it off-line to generate your augmented dataset. An easy to use wrapper for off-line augmentation is provided.

## Installation
tfAugmentor is written in Python and can be easily installed via:
```python
pip install tfAugmentor
```
To run tfAugmentor properly, the following library should be installed as well:
- tensorflow (developed under tf 1.12)
- numpy (developed under numpy 1.15)

## Quick Start
tfAugmentor aims to implement image augmentations purly as a tensorflow graph, so that it can be used seamlessly with other tensorflow components, such as tf.Data. 
But you can also use it independently as a off-line augmentation tool.   

To begin, instantiate an `Augmentor` object and pass a dictionary of tensors to it. These tensors should have the same 4-D shape of `[batch, height, width, channels]`. 

To preserve the consistence of label/segmentation maps, the corresponding dict key should be pass to `label` as a list.

```python
import tfAugmentor as tfa
tensor_list = {
	'images': image_tensor,
	'segmentation_mask': mask_tensor
}
a = tfa.Augmentor(tensor_list, label=['segmentation_mask'])
```

### Use with tf.Data

An example of data importing with tf.data and tfAugmentor:

```
ds = tf.data.TFRecordDataset([...])
ds = ds.map(extract_fn)
ds = ds.shuffle(buffer_size=500)
ds = ds.batch(batch_size)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


def exttact_fn(sample):
	// parse the tfrecord example of your dataset
	....
	
	// assume the dataset contains three tensors: image, weight_map, seg_mask
	
	// instantiate an Augmentor
	
	input_list = {'img': image, 
	'weight': weight_map, 
	'mask': seg_mask}
	a = tfa.Augmentor(input_list, label=['segmentation_mask'])
	
	// apply left right flip with probability 0.5
	a.flip_left_right(probability=0.5)
	// apply random rotation with probability 0.6
	a.random_rotate(probability=0.6)
	// apply elastic deformation
	a.elastic_deform(probability=0.2, strength=200, scale=20)
	
	// dictionary of the augmented images, which has the same keys as input_list
	augmented = a.out
	// return tensors in a form you need
	return augmented['img'], augmented['weight'], augmented['mask'] 
```

### Off-line augmentation

 

## Main Features

### Mirroring
```python
a.flip_left_right(probability) // flip the image left right  
a.flip_up_down(probability) // flip the image up down
```
### Rotating
```python
a.rotate90(probability) // rotate by 90 degree clockwise
a.rotate180(probability) // rotate by 180 degree clockwise
a.rotate270(probability) // rotate by 270 degree clockwise
a.rotate(probability, angle) // rotate by *angel* degree clockwise
a.random_rotate(probability) // randomly rotate the image
```
### crop and resize
```python
a.random_crop_resize(probability, scale_range=(0.5, 0.8)) // randomly crop a sub-image and resize to the same size of the original image
a.crop(probability, size) // randomly crop a sub-image of a certain size
```

### elastic deformation
a.elastic_deform(probability, strength, scale)
