
import tensorflow as tf

def create_weights(shape,name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def create_biases(size,name):
    return tf.Variable(tf.constant(0.1, shape=[size]), name=name)


def conv_Layer(input,
               conv_filter_size,
               num_input_channels,
               num_filters,
               stride_x,
               stride_y,
               scope_name):
    with tf.name_scope(scope_name):

        weight = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters], name="weight")

        biase = create_biases(num_filters, "biase")

        layer = tf.nn.conv2d(input=input,
                             filter=weight,
                             strides=[1, stride_y, stride_x, 1],
                             padding='SAME')
        layer = tf.nn.bias_add(layer, biase)
    return layer

"""max-pooling"""
def max_pool_layer(x,scope_name, kHeight=2, kWidth=2, stride_x=2, stride_y=2, padding = "SAME"):
    with tf.name_scope(scope_name):
        layer =  tf.nn.max_pool(x, ksize=[1, kHeight, kWidth, 1],
                          strides=[1, stride_x, stride_y, 1], padding = padding)
    return layer




def flatten_layer(layer):

    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer


def fc_layer(input,
            num_inputs,
            num_outputs,
            scope_name,
            keep_prob=1,
            use_relu=True):
    with tf.name_scope(scope_name):
        # Let's define trainable weights and biases.
        weights = create_weights(shape=[num_inputs, num_outputs],name='weight')
        biases = create_biases(num_outputs,name='biase')
        layer = tf.matmul(input, weights) + biases
        layer = tf.nn.dropout(layer, keep_prob)
        if use_relu:
            layer = tf.nn.relu(layer)

    return layer

def get_conv_layer(input, weights, biases,use_relu=True,use_max_pooling = True):
    layer = tf.nn.conv2d(input, weights, strides=(1, 1, 1, 1),padding='SAME')
    layer =  tf.nn.bias_add(layer, biases)
    if use_relu:
        layer = tf.nn.relu(layer)
    if use_max_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    return layer
def get_fc_layer(input, weights, biases,use_relu=True):
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer
