from tensorflow.keras import Model
from model.layers import MLP, LogistRegression, SparseInput, Concatenate, Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import L2
import tensorflow as tf


class DCN(Model):
    def __init__(self, dense_feats, sparse_feats, sparse_dims, cross_layer_num=3, embed_dim=8,
                 hide_units=[256, 128, 64], l2_reg=0.01, use_drop=True, drop_rate=0.5, use_bn=False):
        super(DCN, self).__init__()
        self.dense_feats = dense_feats
        self.sparse_feats = sparse_feats
        self.embed_dim = embed_dim
        self.sparse_dims = sparse_dims
        self.cross_layer_num = cross_layer_num
        self.deep = MLP(hide_units, l2_reg, use_drop, drop_rate, use_bn)
        self.cross = CrossNet(cross_layer_num)
        self.sparse_input_layer = SparseInput(sparse_feats, sparse_dims, embed_dim, l2_reg)
        self.Concatenate1 = Concatenate(name='dnn_input')
        self.Concatenate2 = Concatenate(name='add_output')
        self.sigmoid_output = Dense(1, activation='sigmoid')

    def call(self, inputs):
        dense_inputs = inputs[:, :len(self.dense_feats)]
        sparse_inputs = inputs[:, len(self.dense_feats):]
        embed_inputs = self.sparse_input_layer(sparse_inputs)

        share_input = self.Concatenate1([dense_inputs, embed_inputs])
        dnn_output = self.deep(share_input)
        cross_output = self.cross(share_input)

        output = self.Concatenate2([dnn_output, cross_output])
        output = self.sigmoid_output(output)
        return output


class CrossNet(Layer):
    def __init__(self, cross_layer_num=3, w_regular=0.001, b_regular=0.001, use=True):
        super(CrossNet, self).__init__()
        self.cross_layer_num = cross_layer_num
        self.w_regular = w_regular
        self.b_regular = b_regular
        self.use = use

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        assert feat_dim >= 2, 'The feature dimension should not below  2'
        self.cross_weights = [
            self.add_weight(name='w_' + str(i),
                            shape=(feat_dim, 1),
                            initializer='random_normal',
                            regularizer=L2(self.w_regular),
                            trainable=True
                            )
            for i in range(self.cross_layer_num)]
        self.cross_bias = [
            self.add_weight(name='b_' + str(i),
                            shape=(feat_dim, 1),
                            initializer='random_normal',
                            regularizer=L2(self.b_regular),
                            trainable=True
                            )
            for i in range(self.cross_layer_num)]

    def call(self, inputs):
        if self.use == True:
            x_0 = tf.expand_dims(inputs, axis=2)  # (batch_size, dim, 1)  由于有batch_size,为了张量矩阵乘法计算才增加一维。
            x_l = x_0  # (None, dim, 1)
            for i in range(self.cross_layer_num):
                # cross_weights shape = (dim,1)
                x_l1 = tf.tensordot(x_l, self.cross_weights[i], axes=[1, 0])  # (batch_size, dim, dim)
                x_l = tf.matmul(x_0, x_l1) + self.cross_bias[i] + x_l  # (batch_size, dim, 1)
            x_l = tf.squeeze(x_l, axis=2)  # (batch_size, dim)
        else:
            x_0 = inputs  # (batch_size, dim,)
            x_l = x_0  # (batch_size, dim,)

        return x_l
