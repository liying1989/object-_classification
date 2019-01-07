import nets.layer as layer

'''
layers = (
    'layer_conv1_1','layer_conv1_2','layer_conv2_1','layer_conv2_2','layer_conv3_1','layer_conv3_2','layer_conv3_3',
    'layer_conv4_1','layer_conv4_2','layer_conv4_3','layer_conv5_1','layer_conv5_2','layer_conv5_2',
    'layer_fc1','layer_fc2','layer_fc3')
'''


class vgg16(object):
    def __init__(self,x_input,out_class_num):
        self.x = x_input
        self.net_name = 'vgg16'
        self.num_channels = int(self.x.get_shape()[-1])
        self.filter_size_conv_1 = 1
        self.filter_size_conv_3 = 3
        self.num_filters_conv_64 = 64
        self.num_filters_conv_128 = 128
        self.num_filters_conv_256 = 256
        self.num_filters_conv_512 = 512
        self.num_classes = out_class_num
        self.fullConnectClass = 500
        self.fullConnectClass2 = 100
    def num_classes(self):
        return self.num_classes

    def buildModel(self):
        layer_conv1_1 = layer.conv_Layer(input=self.x,
                                         conv_filter_size=self.filter_size_conv_3,
                                         num_input_channels=self.num_channels,
                                         num_filters=self.num_filters_conv_64,
                                         stride_x = 1,
                                         stride_y = 1,
                                         scope_name ='conv1_1')


        layer_conv1_2 = layer.conv_Layer(input=layer_conv1_1,
                                         conv_filter_size=self.filter_size_conv_3,
                                         num_input_channels=self.num_filters_conv_64,
                                         num_filters=self.num_filters_conv_64,
                                         stride_x = 1,
                                         stride_y = 1,
                                         scope_name='conv1_2')

        layer_pool_1 = layer.max_pool_layer(layer_conv1_2,'pool_1')

        layer_conv2_1 = layer.conv_Layer(input=layer_pool_1,
                                         conv_filter_size=self.filter_size_conv_3,
                                         num_input_channels=self.num_filters_conv_64,
                                         num_filters=self.num_filters_conv_128,
                                         stride_x=1,
                                         stride_y=1,
                                         scope_name='conv2_1'
                                         )

        layer_conv2_2 = layer.conv_Layer(input=layer_conv2_1,
                                         conv_filter_size=self.filter_size_conv_3,
                                         num_input_channels=self.num_filters_conv_128,
                                         num_filters=self.num_filters_conv_128,
                                         stride_x=1,
                                         stride_y=1,
                                         scope_name='conv2_2'
                                         )

        layer_pool_2 = layer.max_pool_layer(layer_conv2_2,'pool_2')

        layer_conv3_1 = layer.conv_Layer(input=layer_pool_2,
                                         conv_filter_size=self.filter_size_conv_3,
                                         num_input_channels=self.num_filters_conv_128,
                                         num_filters=self.num_filters_conv_256,
                                         stride_x = 1,
                                         stride_y = 1,
                                         scope_name='conv3_1')

        layer_conv3_2 = layer.conv_Layer(input=layer_conv3_1,
                                         conv_filter_size=self.filter_size_conv_3,
                                         num_input_channels=self.num_filters_conv_256,
                                         num_filters=self.num_filters_conv_256,
                                         stride_x=1,
                                         stride_y=1,
                                         scope_name='conv3_2'
                                         )

        layer_conv3_3 = layer.conv_Layer(input=layer_conv3_2,
                                         conv_filter_size=self.filter_size_conv_3,
                                         num_input_channels=self.num_filters_conv_256,
                                         num_filters=self.num_filters_conv_256,
                                         stride_x=1,
                                         stride_y=1,
                                         scope_name='conv3_3'
                                         )

        layer_pool_3 = layer.max_pool_layer(layer_conv3_3,'pool_3')

        layer_conv4_1 = layer.conv_Layer(input=layer_pool_3,
                                         conv_filter_size=self.filter_size_conv_3,
                                         num_input_channels=self.num_filters_conv_256,
                                         num_filters=self.num_filters_conv_512,
                                         stride_x=1,
                                         stride_y=1,
                                         scope_name='conv4_1'
                                         )
        layer_conv4_2 = layer.conv_Layer(input=layer_conv4_1,
                                         conv_filter_size=self.filter_size_conv_3,
                                         num_input_channels=self.num_filters_conv_512,
                                         num_filters=self.num_filters_conv_512,
                                         stride_x = 1,
                                         stride_y = 1,
                                         scope_name='conv4_2')
        layer_conv4_3 = layer.conv_Layer(input=layer_conv4_2,
                                         conv_filter_size=self.filter_size_conv_3,
                                         num_input_channels=self.num_filters_conv_512,
                                         num_filters=self.num_filters_conv_512,
                                         stride_x=1,
                                         stride_y=1,
                                         scope_name='conv4_3')


        layer_pool_4 = layer.max_pool_layer(layer_conv4_3,'pool_4')

        layer_conv5_1 = layer.conv_Layer(input=layer_pool_4,
                                         conv_filter_size=self.filter_size_conv_3,
                                         num_input_channels=self.num_filters_conv_512,
                                         num_filters=self.num_filters_conv_512,
                                         stride_x=1,
                                         stride_y=1,
                                         scope_name='conv5_1')

        layer_conv5_2 = layer.conv_Layer(input=layer_conv5_1,
                                         conv_filter_size=self.filter_size_conv_3,
                                         num_input_channels=self.num_filters_conv_512,
                                         num_filters=self.num_filters_conv_512,
                                         stride_x=1,
                                         stride_y=1,
                                         scope_name='conv5_2')

        layer_conv5_3 = layer.conv_Layer(input=layer_conv5_2,
                                         conv_filter_size=self.filter_size_conv_3,
                                         num_input_channels=self.num_filters_conv_512,
                                         num_filters=self.num_filters_conv_512,
                                         stride_x=1,
                                         stride_y=1,
                                         scope_name='conv5_3'
                                         )

        layer_pool_5 = layer.max_pool_layer(layer_conv5_3,'pool_5')

        layer_flat = layer.flatten_layer(layer_pool_5)
        fc1 = layer.fc_layer(input=layer_flat,
                             num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                             num_outputs=self.fullConnectClass,
                             scope_name='fc1')
        fc2 = layer.fc_layer(input=fc1, num_inputs=self.fullConnectClass, num_outputs=self.fullConnectClass2, scope_name= 'fc2',keep_prob = 1)
        fc3 = layer.fc_layer(input=fc2, num_inputs=self.fullConnectClass2, num_outputs=self.num_classes, scope_name='fc3',keep_prob = 1)
        return fc3
