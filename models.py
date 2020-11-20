import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, Input, Lambda, MaxPooling2D,
                          Multiply, ReLU, Reshape, Softmax)
from keras.models import Model
import logging
from keras.optimizers import Adam
from keras.applications import imagenet_utils

if not tf.__version__.startswith("2.2") or not keras.__version__.startswith("2.4.3"):
    print("This code was written with TensorFlow 2.2 and Keras 2.4.3, and may fail on your version:")
print("tf:", tf.__version__)
print("keras:", keras.__version__)

MAX_FRET = 5

def xception(inpt):
    """
    keras xception network. see https://keras.io/api/applications/
    """
    base = keras.applications.Xception(include_top=False, weights=None, 
                input_shape=inpt.shape)
    x = base(inpt)
    x = GlobalAveragePooling2D()(x)

    x = Dense(256)(x)
    x = ReLU()(x)
    x = Dropout(0.4)(x)

    return x

def mobilenetv2(inpt):
    """
    keras mobilenetv2
    """
    base = keras.applications.MobileNetV2(include_top=False, weights=None,
                input_shape=inpt.shape, alpha=1.0)
    # x = keras.applications.mobilenet_v2.preprocess_input(inpt)
    x = base(inpt)
    # x = GlobalAveragePooling2D()(x)

    x = Dense(256)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)

    return x



def my_mobilenet(inpt, alpha=0.5, weights='imagenet'):
    """
    https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/applications/mobilenet_v2.py#L496
    """

    input_tensor = inpt

    _batch, rows, cols, channels = input_tensor.shape

    if weights == 'imagenet':
        if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
            raise ValueError('If imagenet weights are being loaded, '
                            'alpha can be one of `0.35`, `0.50`, `0.75`, '
                            '`1.0`, `1.3` or `1.4` only.')

        if rows != cols or rows not in [96, 128, 160, 192, 224]:
            rows = 224
            logging.warning('`input_shape` is undefined or non-square, '
                        'or `rows` is not in [96, 128, 160, 192, 224].'
                        ' Weights for input shape (224, 224) will be'
                        ' loaded as the default.')


    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.ZeroPadding2D(
        padding=correct_pad(input_tensor, 3),
        name='Conv1_pad')(input_tensor)
    x = layers.Conv2D(
        first_block_filters,
        kernel_size=3,
        strides=(2, 2),
        padding='valid',
        use_bias=False,
        name='Conv1')(
            x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(
            x)
    x = layers.ReLU(6., name='Conv1_relu')(x)

    x = _inverted_res_block(
        x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

    x = _inverted_res_block(
        x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
    x = _inverted_res_block(
        x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

    x = _inverted_res_block(
        x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    x = _inverted_res_block(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    x = _inverted_res_block(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

    x = _inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
    x = _inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
    x = _inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

    x = _inverted_res_block(
        x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)
    x = _inverted_res_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
    x = _inverted_res_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)

    x = _inverted_res_block(
        x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    # these layers actually get scrapped; they are just so the model size works to load imagenet weights
    x = layers.Conv2D(
        last_block_filters, kernel_size=1, use_bias=False, name='Conv_1')(
            x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(
            x)
    x = layers.ReLU(6., name='out_relu')(x)


    # Create model.
    weightsmodel = Model(input_tensor, x, name='mobilenetv2_%0.2f_%s' % (alpha, rows))

    # Load weights.
    if weights == 'imagenet':
        model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' +
                        str(alpha) + '_' + str(rows) + '_no_top' + '.h5')
        BASE_WEIGHT_PATH = ('https://storage.googleapis.com/tensorflow/'
                            'keras-applications/mobilenet_v2/')
        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = keras.utils.get_file(
            model_name, weight_path, cache_subdir='models')
        weightsmodel.load_weights(weights_path)
    elif weights is not None:
        weightsmodel.load_weights(weights)

    real_output = weightsmodel.layers[-4].output

    x = layers.Conv2D(64, kernel_size=(2,4), use_bias=False, name='Conv_1')(real_output)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
    x = layers.ReLU(6., name='relu_1')(x)

    x = layers.Conv2D(16, kernel_size=(3,3), use_bias=False, name='Conv_2')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv_2_bn')(x)
    x = layers.ReLU(6., name='out_relu')(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = ReLU(6.)(x)

    return x


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
  """Inverted ResNet block."""
  channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

  in_channels = K.int_shape(inputs)[channel_axis]
  pointwise_conv_filters = int(filters * alpha)
  pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
  x = inputs
  prefix = 'block_{}_'.format(block_id)

  if block_id:
    # Expand
    x = layers.Conv2D(
        expansion * in_channels,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None,
        name=prefix + 'expand')(
            x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'expand_BN')(
            x)
    x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
  else:
    prefix = 'expanded_conv_'

  # Depthwise
  if stride == 2:
    x = layers.ZeroPadding2D(
        padding=correct_pad(x, 3),
        name=prefix + 'pad')(x)
  x = layers.DepthwiseConv2D(
      kernel_size=3,
      strides=stride,
      activation=None,
      use_bias=False,
      padding='same' if stride == 1 else 'valid',
      name=prefix + 'depthwise')(
          x)
  x = layers.BatchNormalization(
      axis=channel_axis,
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'depthwise_BN')(
          x)

  x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

  # Project
  x = layers.Conv2D(
      pointwise_filters,
      kernel_size=1,
      padding='same',
      use_bias=False,
      activation=None,
      name=prefix + 'project')(
          x)
  x = layers.BatchNormalization(
      axis=channel_axis,
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'project_BN')(
          x)

  if in_channels == pointwise_filters and stride == 1:
    return layers.Add(name=prefix + 'add')([inputs, x])
  return x


def _make_divisible(v, divisor, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    Arguments:
        inputs: Input tensor.
        kernel_size: An integer or tuple/list of 2 integers.
    Returns:
        A tuple.
    """
    img_dim = 2 if K.image_data_format() == 'channels_first' else 1
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]),
          (correct[1] - adjust[1], correct[1]))



def make_model(name, input_shape, output_confidences):
    """
    get model from case insensitive name
    args:
        model name
        input shape (not including batch size)
        output_confidences (bool): whether loss to output a prediction, or a 
            softmax confidence for each string
    """
    inpt = Input(input_shape)

    name = name.lower()
    if name == "xception":
        x = xception(inpt)
    elif name == "mobilenetv2":
        x = mobilenetv2(inpt)
    elif name == "mymobilenet":
        x = my_mobilenet(inpt)
    else:
        raise ValueError("no model named '" + name + "'")


    # all models output a vector of size 256
    if output_confidences:
        x = Dense(64)(x)
        x = ReLU()(x)
        x = Dropout(0.4)(x)

        # if the maximum fret is 5, there are 6 options, because 0 is a possibility
        NUM_FRETS = MAX_FRET + 1
        x = Dense(6 * NUM_FRETS)(x)
        x = Reshape((6,NUM_FRETS))(x)
        x = Softmax(axis=-1)(x)

    else:
        x = Dense(32)(x)
        x = ReLU()(x)
        x = Dropout(0.4)(x)

        x = Dense(6)(x)
        # sigmoid limits to between 0 and 1. Then we multiply by the constant to
        # allow a range from 0 to MAX_FRET to be predicted for each string
        x = Activation('sigmoid')(x)
        x = Lambda(lambda v: MAX_FRET * v)(x)

    return Model(inpt, x)




