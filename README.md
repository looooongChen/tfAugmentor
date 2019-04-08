
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
```python
import tfAugmentor as tfa
tensor_list = {
	'images': image_tensor,
	'segmentation_mask': mask_tensor
}
a = tfa.tfAugmentor(tensor_list, label=['segmentation_mask'])
```

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