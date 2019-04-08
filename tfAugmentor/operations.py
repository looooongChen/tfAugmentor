from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops


def _interpolate(grid, query_points,
                 interpolation='bilinear',
                 name='interpolate'):
    """Similar to Matlab's interp2 function.
    Finds values for query points on a grid using bilinear interpolation.
    Args:
    grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
    query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
    name: a name for the operation (optional).
    indexing: whether the query points are specified as row and column (ij),
      or Cartesian coordinates (xy).
    Returns:
    values: a 3-D `Tensor` with shape `[batch, N, channels]`
    Raises:
    ValueError: if the indexing mode is invalid, or if the shape of the inputs
      invalid.
    """

    with ops.name_scope(name):
        grid = ops.convert_to_tensor(grid)
        query_points = ops.convert_to_tensor(query_points)
        shape = grid.get_shape().as_list()
        if len(shape) != 4:
            msg = 'Grid must be 4 dimensional. Received size: '
            raise ValueError(msg + str(grid.get_shape()))

        batch_size, height, width, channels = (array_ops.shape(grid)[0],
                                               array_ops.shape(grid)[1],
                                               array_ops.shape(grid)[2],
                                               array_ops.shape(grid)[3])

        shape = [batch_size, height, width, channels]
        query_type = query_points.dtype
        grid_type = grid.dtype

        with ops.control_dependencies([
            check_ops.assert_equal(
                len(query_points.get_shape()),
                3,
                message='Query points must be 3 dimensional.'),
            check_ops.assert_equal(
                array_ops.shape(query_points)[2],
                2,
                message='Query points must be size 2 in dim 2.')
        ]):
            num_queries = array_ops.shape(query_points)[1]

        with ops.control_dependencies([
            check_ops.assert_greater_equal(
                height, 2, message='Grid height must be at least 2.'),
            check_ops.assert_greater_equal(
                width, 2, message='Grid width must be at least 2.')
        ]):
            alphas = []
            floors = []
            ceils = []
            unstacked_query_points = array_ops.unstack(query_points, axis=2)

        for dim in [0, 1]:
            with ops.name_scope('dim-' + str(dim)):
                queries = unstacked_query_points[dim]

                size_in_indexing_dimension = shape[dim + 1]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = math_ops.cast(size_in_indexing_dimension - 2, query_type)
                min_floor = constant_op.constant(0.0, dtype=query_type)
                floor = math_ops.minimum(
                    math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
                int_floor = math_ops.cast(floor, dtypes.int32)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = math_ops.cast(queries - floor, grid_type)
                min_alpha = constant_op.constant(0.0, dtype=grid_type)
                max_alpha = constant_op.constant(1.0, dtype=grid_type)
                alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = array_ops.expand_dims(alpha, 2)
                alphas.append(alpha)

        with ops.control_dependencies([
            check_ops.assert_less_equal(
                math_ops.cast(batch_size * height * width, dtype=dtypes.float32),
                np.iinfo(np.int32).max / 8,
                message="""The image size or batch size is sufficiently large
                           that the linearized addresses used by array_ops.gather
                           may exceed the int32 limit.""")
        ]):
            flattened_grid = array_ops.reshape(
              grid, [batch_size * height * width, channels])
            batch_offsets = array_ops.reshape(
              math_ops.range(batch_size) * height * width, [batch_size, 1])

        # This wraps array_ops.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using array_ops.gather_nd.
        def gather(y_coords, x_coords, name):
            with ops.name_scope('gather-' + name):
                linear_coordinates = batch_offsets + y_coords * width + x_coords
                gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
                return array_ops.reshape(gathered_values,
                                         [batch_size, num_queries, channels])

        # grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1], 'top_left')
        top_right = gather(floors[0], ceils[1], 'top_right')
        bottom_left = gather(ceils[0], floors[1], 'bottom_left')
        bottom_right = gather(ceils[0], ceils[1], 'bottom_right')

        # now, do the actual interpolation
        with ops.name_scope('interpolate'):
            if interpolation == 'knearest':
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


def warp_image(image, flow, interpolation="bilinear", name='dense_image_warp'):
    """Image warping using per-pixel flow vectors.
    Args:
    image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
    flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
    name: A name for the operation (optional).
    Note that image and flow can be of type tf.half, tf.float32, or tf.float64,
    and do not necessarily have to be the same type.
    Returns:
    A 4-D float `Tensor` with shape`[batch, height, width, channels]`
      and same type as input image.
    Raises:
    ValueError: if height < 2 or width < 2 or the inputs have the wrong number
                of dimensions.
    """
    type = image.dtype

    with ops.name_scope(name):
        batch_size, height, width, channels = (array_ops.shape(image)[0],
                                               array_ops.shape(image)[1],
                                               array_ops.shape(image)[2],
                                               array_ops.shape(image)[3])

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y = array_ops.meshgrid(math_ops.range(width), math_ops.range(height))
    stacked_grid = math_ops.cast(array_ops.stack([grid_y, grid_x], axis=2), flow.dtype)
    batched_grid = array_ops.expand_dims(stacked_grid, axis=0)
    query_points_on_grid = batched_grid - flow
    query_points_flattened = array_ops.reshape(query_points_on_grid,
                                               [batch_size, height * width, 2])
    # Compute values at the query points, then reshape the result back to the
    # image grid.
    image_float = tf.cast(image, tf.float32)
    interpolated = _interpolate(image_float, query_points_flattened,
                                interpolation=interpolation)
    interpolated = array_ops.reshape(interpolated,
                                     [batch_size, height, width, channels])
    return tf.cast(interpolated, type)


def gaussian_kernel(mean, std):
    size = round(std*3)
    d = tf.distributions.Normal(mean/1.0, std/1.0)
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
