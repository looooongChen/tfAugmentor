from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np

try:
    import tensorflow_probability as tfp
    tfp_found = True
except:
    tfp_found = False

try:
    import tensorflow_addons as tfa
    tfa_found = True
except:
    tfa_found = False

#### runner classes ####

class Sync(metaclass=ABCMeta):

    def __init__(self, probability=None):
        self.probability = probability
        self.Ops = {}

    def sync(self, name, op_obj):
        self.Ops[name] = op_obj

    @abstractmethod
    def run(self):
        pass

class Runner(Sync):

    def run(self, images):
        occur = tf.random.uniform([], 0, 1) < self.probability 
        for k in images.keys():
            if k in self.Ops.keys():
                images[k] = self.Ops[k].run(images[k], occur)

class RandomRotateRunner(Sync):

    def run(self, images):
        occur = tf.random.uniform([], 0, 1) < self.probability 
        angle = tf.random.uniform([1], 0, 2*3.1415926)
        for k in images.keys():
            if k in self.Ops.keys():
                images[k] = self.Ops[k].run(images[k], angle, occur)

class RandomCropRunner(Sync):
    
    def __init__(self, probability, scale):
        super().__init__(probability)
        if isinstance(scale, tuple) or isinstance(scale, list):
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def get_bbx(self, batch_size):
        sz = tf.random.uniform([batch_size, 2], self.scale[0], self.scale[1])
        offset = tf.multiply(1-sz, tf.random.uniform([batch_size, 2], 0, 1))
        return tf.concat([offset, offset+sz], axis=1)

    def run(self, images):
        img = images[list(self.Ops.keys())[0]]
        batch_size = 1 if tf.size(tf.shape(img)) == 3 else tf.shape(img)[0]

        occur = tf.random.uniform([], 0, 1) < self.probability 
        bbx = self.get_bbx(batch_size)
        for k in images.keys():
            if k in self.Ops.keys():
                images[k] = self.Ops[k].run(images[k], bbx, occur)

class ElasticDeformRunner(Sync):

    def __init__(self, probability, strength, scale):
        super().__init__(probability)
        self.strength = strength * 100
        self.scale = scale

    def get_flow(self, size):

        ''' size: shape of a 3D or 4D tensor '''

        expand = tf.size(size) != 4
        batch = 1 if expand else size[0] 

        dx = tf.random.uniform([batch,
                                tf.math.floordiv(size[-3], self.scale),
                                tf.math.floordiv(size[-2], self.scale),
                                1], -1, 1)
        dy = tf.random.uniform([batch,
                                tf.math.floordiv(size[-3], self.scale),
                                tf.math.floordiv(size[-2], self.scale),
                                1], -1, 1)
        dx = gaussian(dx, 0, 5)
        dy = gaussian(dy, 0, 5)
        flow = self.strength * tf.concat([dx, dy], axis=-1)
        # flow = tf.image.resize(flow, size[-3:-1])
        flow = reisz_image(flow, size[-3:-1], interpolation='bilinear')

        return flow

    def run(self, images):
        img_sz = tf.shape(images[list(self.Ops.keys())[0]])

        occur = tf.random.uniform([], 0, 1) < self.probability 
        flow = self.get_flow(img_sz)
        for k in images.keys():
            if k in self.Ops.keys():
                images[k] = self.Ops[k].run(images[k], flow, occur)

#### opeartion classed ####

class Op(metaclass=ABCMeta):

    @abstractmethod
    def run(self):
        pass

class LeftRightFlip(Op):

    def run(self, image, occur):
        expand = tf.size(tf.shape(image)) != 4
        image = tf.expand_dims(image, axis=0) if expand else image
        image =  tf.image.flip_left_right(image) if occur else image
        image = tf.squeeze(image, axis=0) if expand else image
        return image


class UpDownFlip(Op):
    def run(self, image, occur):
        expand = tf.size(tf.shape(image)) != 4
        image = tf.expand_dims(image, axis=0) if expand else image
        image = tf.image.flip_up_down(image) if occur else image
        image = tf.squeeze(image, axis=0) if expand else image
        return image

class Rotate90(Op):

    def run(self, image, occur):
        return tf.image.rot90(image, 1) if occur else image

class Rotate180(Op):
    def run(self, image, occur):
        return tf.image.rot90(image, 2) if occur else image

class Rotate270(Op):

    def run(self, image, occur):
        return tf.image.rot90(image, 3) if occur else image

class Rotate(Op):
    def __init__(self, angle, interpolation='bilinear'):
        self.angle = angle
        self.interpolation = interpolation

    def run(self, image, occur):
        expand = tf.size(tf.shape(image)) != 4
        image = tf.expand_dims(image, axis=0) if expand else image
        image = tfa.image.rotate(image, self.angle, interpolation=self.interpolation.upper()) if occur else image
        image = tf.squeeze(image, axis=0) if expand else image
        return image

class RandomRotate(Op):

    def __init__(self, interpolation='bilinear'):
        self.interpolation = interpolation

    def run(self, image, angle, occur):
        expand = tf.size(tf.shape(image)) != 4
        image = tf.expand_dims(image, axis=0) if expand else image
        image = tfa.image.rotate(image, angle, interpolation=self.interpolation.upper()) if occur else image
        image = tf.squeeze(image, axis=0) if expand else image
        return image

class RandomCrop(Op):

    def __init__(self, interpolation='bilinear'):
        self.interpolation = interpolation

    def run(self, image, bbx, occur):
        expand = tf.size(tf.shape(image)) != 4
        batch_size = 1 if expand else tf.shape(image)[0]
        bbx_ind = tf.range(0, batch_size, delta=1, dtype=tf.int32)

        image = tf.expand_dims(image, axis=0) if expand else image
        image = tf.cast(tf.image.crop_and_resize(image, bbx, bbx_ind, tf.shape(image)[1:3], method=self.interpolation.lower()), image.dtype) if occur else image
        image = tf.squeeze(image, axis=0) if expand else image
        return image

class ElasticDeform(Op):

    def __init__(self, interpolation='bilinear'):
        self.interpolation = interpolation

    def run(self, image, flow, occur):
        expand = tf.size(tf.shape(image)) != 4
        image = tf.expand_dims(image, axis=0) if expand else image
        image = warp_image(image, flow, interpolation=self.interpolation.lower()) if occur else image
        image = tf.squeeze(image, axis=0) if expand else image

        return image

def reisz_image(image, size, interpolation='bilinear'):
        ''' 
        Args:
            image: shape of B x H x W x C or H x W x C 
            size: 1D with 2 elements (newH, newW)
        '''

        if tf.reduce_all(tf.shape(image)[-3:-1] == size):
            return image

        expand = tf.size(tf.shape(image)) != 4
        batch_size = 1 if expand else tf.shape(image)[0]
        H, W, channels = tf.shape(image)[-3], tf.shape(image)[-2], tf.shape(image)[-1] 
        if expand:
            image = tf.expand_dims(image, axis=0)
        newH, newW = size[0], size[1]
        
        # get the query coordinates
        grid_x, grid_y = tf.meshgrid(tf.range(newW), tf.range(newH))
        grid_x = tf.cast(grid_x, tf.float32) * tf.cast(W / newW, tf.float32)
        grid_y = tf.cast(grid_y, tf.float32) * tf.cast(H / newH, tf.float32)
        stacked_grid = tf.stack([grid_y, grid_x], axis=2)
        batched_grid = tf.expand_dims(stacked_grid, axis=0)
        query_points_on_grid = batched_grid
        query_points_flattened = tf.reshape(query_points_on_grid, [batch_size, newH * newW, 2])
        # Compute values at the query points, then reshape the result back to the image grid.
        image_float = tf.cast(image, tf.float32)
        interpolated = interpolate(image_float, query_points_flattened, interpolation=interpolation)
        interpolated = tf.reshape(interpolated, [batch_size, newH, newW, channels])
        if expand:
            interpolated = tf.squeeze(interpolated, axis=0)

        return tf.cast(interpolated, image.dtype)

def interpolate(grid, query_points, interpolation='bilinear'):
    """
    Similar to Matlab's interp2 function.
    Finds values for query points on a grid using bilinear interpolation.
    
    Args:
        grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
        query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
        interpolation: interpolation method 'bilinear' or 'nearest'

    Returns:
        values: a 3-D `Tensor` with shape `[batch, N, channels]`
    
    Raises:
        ValueError: if the indexing mode is invalid, or if the shape of the inputs invalid.
    """

    shape = tf.shape(grid)
    batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    num_queries = tf.shape(query_points)[1]

    query_type = query_points.dtype
    grid_type = grid.dtype

    alphas = []
    floors = []
    ceils = []
    unstacked_query_points = tf.unstack(query_points, axis=2)

    for dim in [0, 1]:
        # with ops.name_scope('dim-' + str(dim)):
        queries = unstacked_query_points[dim]
        size_in_indexing_dimension = shape[dim + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1 is still a valid index into the grid.
        max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
        min_floor = tf.constant(0.0, dtype=query_type)
        floor = tf.minimum(tf.maximum(min_floor, tf.floor(queries)), max_floor)
        int_floor = tf.cast(floor, tf.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # compute alpha for taking linear combinations of pixel values from the image
        # same dtype with grid
        alpha = tf.cast(queries - floor, grid_type)
        min_alpha = tf.constant(0.0, dtype=grid_type)
        max_alpha = tf.constant(1.0, dtype=grid_type)
        alpha = tf.minimum(tf.maximum(min_alpha, alpha), max_alpha)
        alpha = tf.expand_dims(alpha, 2)
        alphas.append(alpha)

    flattened_grid = tf.reshape(grid, [batch_size * height * width, channels])
    batch_offsets = tf.reshape(tf.range(batch_size) * height * width, [batch_size, 1])

    # helper function to get value from the flattened tensor
    def gather(y_coords, x_coords):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = tf.gather(flattened_grid, linear_coordinates)
        return tf.reshape(gathered_values, [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1])
    top_right = gather(floors[0], ceils[1])
    bottom_left = gather(ceils[0], floors[1])
    bottom_right = gather(ceils[0], ceils[1])

    # now, do the actual interpolation
    if interpolation == 'nearest':
        t = tf.less(alphas[0], 0.5)
        l = tf.less(alphas[1], 0.5)

        tl = tf.cast(tf.logical_and(t, l), tf.float32)
        tr = tf.cast(tf.logical_and(t, tf.logical_not(l)), tf.float32)
        bl = tf.cast(tf.logical_and(tf.logical_not(t), l), tf.float32)
        br = tf.cast(tf.logical_and(tf.logical_not(t), tf.logical_not(l)), tf.float32)

        interp = tf.multiply(top_left, tl) + tf.multiply(top_right, tr) \
                    + tf.multiply(bottom_left, bl) + tf.multiply(bottom_right, br)
    else:
        interp_top = alphas[1] * (top_right - top_left) + top_left
        interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
        interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp


# def warp_image(image, flow, interpolation="bilinear"):
#     """
#     Image warping using per-pixel flow vectors.

#     Args:
#         image: 4-D `Tensor` with shape [batch, height, width, channels]
#         flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
    
#     Note: the image and flow can be of type tf.half, tf.float32, or tf.float64, and do not necessarily have to be the same type.
    
#     Returns:
#         A 4-D float `Tensor` if shape [batch, height, width, channels] with same type as input image.
    
#     Raises:
#         ValueError: if height < 2 or width < 2 or the inputs have the wrong number of dimensions.
#     """
#     sz = tf.shape(image)
#     batch_size, height, width, channels = sz[0], sz[1], sz[2], sz[3]
#     # get the query coordinates
#     grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
#     stacked_grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), flow.dtype)
#     batched_grid = tf.expand_dims(stacked_grid, axis=0)
#     query_points_on_grid = batched_grid - flow
#     query_points_flattened = tf.reshape(query_points_on_grid, [batch_size, height * width, 2])
#     # Compute values at the query points, then reshape the result back to the image grid.
#     image_float = tf.cast(image, tf.float32)
#     interpolated = interpolate(image_float, query_points_flattened, interpolation=interpolation)
#     interpolated = tf.reshape(interpolated, [batch_size, height, width, channels])

#     return tf.cast(interpolated, image.dtype)


def warp_image(image, flow, interpolation="bilinear"):
    """
    Image warping using per-pixel flow vectors.

    Args:
        image: 4-D `Tensor` with shape [batch, height, width, channels]
        flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
    
    Note: the image and flow can be of type tf.half, tf.float32, or tf.float64, and do not necessarily have to be the same type.
    
    Returns:
        A 4-D float `Tensor` if shape [batch, height, width, channels] with same type as input image.
    
    Raises:
        ValueError: if height < 2 or width < 2 or the inputs have the wrong number of dimensions.
    """
    sz = tf.shape(flow)
    batch_size, height, width = sz[0], sz[1], sz[2]
    channels = tf.shape(image)[3]

    # get the query coordinates
    grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
    stacked_grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), flow.dtype)
    batched_grid = tf.expand_dims(stacked_grid, axis=0)
    query_points_on_grid = batched_grid - flow
    query_points_flattened = tf.reshape(query_points_on_grid, [batch_size, height * width, 2])
    # Compute values at the query points, then reshape the result back to the image grid.
    image_float = tf.cast(image, tf.float32)
    interpolated = interpolate(image_float, query_points_flattened, interpolation=interpolation)
    interpolated = tf.reshape(interpolated, [batch_size, height, width, channels])

    return tf.cast(interpolated, image.dtype)

def gaussian_kernel(mean, std):
    size = round(std*3)
    if tfp_found:
        d = tfp.distributions.Normal(mean/1.0, std/1.0)
    else:
        d = tf.compat.v1.distributions.Normal(mean/1.0, std/1.0)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def gaussian(image, mean, std):
    gauss_kernel = gaussian_kernel(mean, std)
    kernel = tf.expand_dims(tf.expand_dims(gauss_kernel, -1), -1)
    return tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding="SAME")

if __name__ == "__main__":
    a = gaussian_kernel(2, 5)
    print(a)
    b = gaussian_kernel(2, 5)
    print(b)
