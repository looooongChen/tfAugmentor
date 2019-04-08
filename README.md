
still under construction, coming soon
# tfAugmentor
An image augmentation library for tensorflow. All operations are implemented as pure tensorflow graph operations. Thus, tfAugmentor can be easily combined with any tensorflow graph, such as tf.Data, for on-the-fly data augmentation. 
Of cousrse, you can also use it off-line to generate your augmented dataset. An easy to use wrapper for off-line augmentation is provided.

## Quick Start
```python
import tfAugmentor
tensor_list = {
	'images': image_tensor,
	'segmentation_mask': mask_tensor
}
a = tfAugmentor(tensor_list, label=['segmentation_mask'])
```

## Main Features

### Mirroring
```python
.flip_left_right(prob)
.flip_up_down(prob)
```
### Rotating
```python
.rotate90(prob)
.rotate180(prob)
.rotate270(prob)
.rotate(prob, angle)
.random_rotate(prob)
```
### crop and resize
```python
.random_crop_resize(prob, scale_range=(0.5, 0.8))
.crop(prob, sz)
```