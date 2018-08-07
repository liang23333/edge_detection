import tensorflow as tf

import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]


class HED:
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path='vgg16.npy', trainable=True, dropout=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout
        self.image = tf.placeholder(tf.float32, [1, None, None, 3], name='input_image')
        self.label = tf.placeholder(tf.float32, [1, None, None, 1], name='input_label')
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        # rgb_scaled = rgb * 255.0
        #
        # # Convert RGB to BGR
        # red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        # bgr = tf.concat(axis=3, values=[
        #     blue - VGG_MEAN[0],
        #     green - VGG_MEAN[1],
        #     red - VGG_MEAN[2],
        # ])
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(self.image, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")

        #first

        self.conv1_end=self.conv_layer(self.conv1_2,64,1,'conv1_end',kernel_size=1)
        self.dsn1=self.conv1_end
        self.out1=tf.sigmoid(self.dsn1)
        self.loss1=self.class_balanced_sigmoid_cross_entropy(logits=self.dsn1,label=self.label)

        #second

        self.conv2_end=self.conv_layer(self.conv2_2,128,1,'conv2_end',kernel_size=1)
        self.dsn2=self.deconv(self.conv2_end,2)
        self.out2=tf.sigmoid(self.dsn2)
        self.loss2=self.class_balanced_sigmoid_cross_entropy(logits=self.dsn2,label=self.label)

        #third
        self.conv3_end=self.conv_layer(self.conv3_3,256,1,'conv3_end',kernel_size=1)
        self.dsn3=self.deconv(self.conv3_end,4)
        self.out3=tf.sigmoid(self.dsn3)
        self.loss3=self.class_balanced_sigmoid_cross_entropy(logits=self.dsn3,label=self.label)

        #fourth
        self.conv4_end=self.conv_layer(self.conv4_3,512,1,'conv4_end',kernel_size=1)
        self.dsn4=self.deconv(self.conv4_end,8)
        self.out4=tf.sigmoid(self.dsn4)
        self.loss4=self.class_balanced_sigmoid_cross_entropy(logits=self.dsn4,label=self.label)


        #fifth
        self.conv5_end=self.conv_layer(self.conv5_3,512,1,'conv5_end',kernel_size=1)
        self.dsn5=self.deconv(self.conv5_end,16)
        self.out5=tf.sigmoid(self.dsn5)
        self.loss5=self.class_balanced_sigmoid_cross_entropy(logits=self.dsn5,label=self.label)


        #concat

        self.Concat=tf.concat([self.dsn1,self.dsn2,self.dsn3,self.dsn4,self.dsn5],axis=3,name='Concat')
        self.dsn=self.conv_layer(self.Concat,5,1,'dsn',kernel_size=1)
        self.out=tf.sigmoid(self.dsn)
        self.loss=self.class_balanced_sigmoid_cross_entropy(logits=self.dsn,label=self.label)


        self.loss_sum=self.loss1+self.loss2+self.loss3+self.loss4+self.loss5+self.loss
        tf.summary.scalar('loss_sum',self.loss_sum)
    def class_balanced_sigmoid_cross_entropy(self,logits, label):
        ## ref - https://github.com/ppwwyyxx/tensorpack/blob/master/examples/HED/hed.py
        """
        The class-balanced cross entropy loss,
        as in `Holistically-Nested Edge Detection
        <http://arxiv.org/abs/1504.06375>`_.
        Args:
            logits: of shape (b, ...).
            label: of the same shape. the ground truth in {0,1}.
        Returns:
            class-balanced cross entropy loss.
        """

        with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
            count_neg = tf.reduce_sum(1.0 - label)  # 样本中0的数量
            count_pos = tf.reduce_sum(label)  # 样本中1的数量(远小于count_neg)
            # print('debug, ==========================, count_pos is: {}'.format(count_pos))
            beta = count_neg / (count_neg + count_pos)  ## e.g.  60000 / (60000 + 800) = 0.9868

            pos_weight = beta / ((1.0 - beta)*2)  ## 0.9868 / (1.0 - 0.9868) = 0.9868 / 0.0132 = 74.75
            cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=label, pos_weight=pos_weight)
            print(cost.shape)
            cost = tf.reduce_mean(cost * ((1 - beta)*2))

            # 如果样本中1的数量等于0，那就直接让 cost 为 0，因为 beta == 1 时， 除法 pos_weight = beta / (1.0 - beta) 的结果是无穷大
            zero = tf.equal(count_pos, 0.0)
            final_cost = tf.where(zero, 0.0, cost)
        return final_cost

    def get_kernel_size(self,factor):
        """
        Find the kernel size given the desired factor of upsampling.
        """
        return 2 * factor - factor % 2

    def upsample_filt(self,size):
        """
        Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
        """
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

    def bilinear_upsample_weights(self,factor, number_of_classes):
        """
        Create weights matrix for transposed convolution with bilinear filter
        initialization.
        """
        filter_size = self.get_kernel_size(factor)

        weights = np.zeros((filter_size,
                            filter_size,
                            number_of_classes,
                            number_of_classes), dtype=np.float32)

        upsample_kernel = self.upsample_filt(filter_size)

        for i in range(number_of_classes):
            weights[:, :, i, i] = upsample_kernel

        return weights


    def deconv(self,inputs, upsample_factor):

        # Calculate the ouput size of the upsampled tensor

        upsample_filter_np = self.bilinear_upsample_weights(upsample_factor, 1)
        upsample_filter_tensor = tf.Variable(upsample_filter_np)

        # Perform the upsampling
        upsampled_inputs = tf.nn.conv2d_transpose(inputs, upsample_filter_tensor,
                                                  output_shape=tf.shape(self.label),
                                                  strides=[1, upsample_factor, upsample_factor, 1])

        return upsampled_inputs




    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name,kernel_size=3):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(kernel_size, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg16-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count