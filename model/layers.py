import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense, Dropout, BatchNormalization,Embedding,Concatenate
from tensorflow.keras.regularizers import L1,L2
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers import Dense


class MLP(Layer):
    def __init__(self, hide_units = [256, 128, 64], l2_reg = 0.01, use_drop = True, drop_rate = 0.5, use_bn = False):
        '''
        :param hide_units:  list, units of MLP
        :param l2_reg: the constraint of L2 regularization, default is 0, mean no use
        :param use_drop: bool, default is True
        :param drop_rate: flot, [0,1], default is 0.5
        :param use_bn: bool, default is False
        return: 输出network的输出，还未归一化 输出shape [64] units[-1]
        '''
        super(MLP, self).__init__()
        assert len(hide_units) != 0, f'unit cannot be empty{hide_units}'
        self.denses = []
        self.dropouts = []
        self.bns = []
        self.l2_reg  = l2_reg
        self.use_drop = use_drop
        self.use_bn = use_bn

        for unit in hide_units:
            self.denses.append( Dense(unit, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer= L2(self.l2_reg)  ))
            self.dropouts.append(Dropout(drop_rate))
            self.bns.append(BatchNormalization())



    def call(self,inputs):
        x = inputs
        for (dense,dropout,bn) in zip(self.denses, self.dropouts,self.bns):
            x = dense(x)
            if self.use_drop:
                x = dropout(x)
            if self.use_bn:
                x = bn(x)
        return x


class LogistRegression(Layer):
    def __init__(self):
        super(LogistRegression, self).__init__()

    def build(self,input_shape):
        input_size = input_shape[-1]
        self.logist = Dense(units = input_size,use_bias=False, activation= None,kernel_initializer='uniform')

    def call(self,inputs):
        output = self.logist(inputs)
        return output

class SparseInput(Layer):
    def __init__(self, sparse_feats, sparse_dims,embed_dim,l2_reg):
        super(SparseInput, self).__init__()
        self.sparse_feats = sparse_feats
        self.embeds = []
        for i,feat  in enumerate(sparse_feats):
            self.embeds.append(
            Embedding(
            input_dim= sparse_dims[feat],
            output_dim= 8,
            embeddings_regularizer= L2(l2_reg),
            name= f'Embedding{feat}'
            )
                )
        self.concatenate = Concatenate()

    def call (self,inputs):
        embeddings = []
        for i in range(len(self.sparse_feats)):
            embeddings.append(self.embeds[i](inputs[:, i]))
        sparse_inputs = self.concatenate(embeddings)
        return sparse_inputs


class SparseHistInput(Layer):
    def __init__(self, sparse_feats, sparse_dims,embed_dim,l2_reg):
        super(SparseHistInput, self).__init__()
        self.sparse_feats = sparse_feats
        self.embeds = []
        for i,feat  in enumerate(sparse_feats):
            self.embeds.append(
            Embedding(
            input_dim= sparse_dims[feat],
            output_dim= 8,
            embeddings_regularizer= L2(l2_reg),
            name= f'Embedding{feat}'
            )
                )
        self.concatenate = Concatenate()

    def call (self,inputs):
        embeddings = []
        for i in range(len(self.sparse_feats)):
            embeddings.append(self.embeds[i](inputs[:, i]))
        sparse_inputs = self.concatenate(embeddings)
        return sparse_inputs






        


def create_embedding_matrix(feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True):
    from . import feature_column as fc_lib

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict


class LocalActivationLayer(Layer):
    """
    DIN 中的attention部分
    """

    def __init__(self, att_units, max_hist_len=20,  l2_reg=0.02):
        super(LocalActivationLayer, self).__init__()
        self.attention_mlp = MLP(hide_units=att_units)
        self.dense_output = Dense(1,activation=None,use_bias =None,name='att_fcn_output')
        self.l2_reg = l2_reg
        self.max_hist_len = max_hist_len


    def call(self, inputs):
        '''
        inputs =(queries, keys, keys_length)
        :param queries: 待预测商品特征
        :param keys: 用户行为的商品特征列表
        :param keys_length: 行为特征的长度,并不一定统一长度。
        :return:
        这里的输入有三个，候选广告queries，用户历史行为keys，以及Batch中每个行为的长度。这里为什么要输入一个keys_length呢，因为每个用户发生过的历史行为是不一样多的，但是输入的keys维度是固定的(都是历史行为最大的长度)，因此我们需要这个长度来计算一个mask，告诉模型哪些行为是没用的，哪些是用来计算用户兴趣分布的。
      Input shape
        - A list of three tensor: [query,keys,keys_length]
        - query is a 3D tensor with shape:  ``(batch_size, embedding_size)``
        - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``
        - keys_length is a 2D tensor with shape: ``(batch_size, 1)``
        '''
        # Attention

        queries, keys, keys_length = inputs
        # 1.将query复制为多份(T 份)，对应每一个keys。
        queries = tf.expand_dims(queries, axis=1)  # ((batch_size, 1, embedding_size))
        queries = tf.tile(queries, [1, tf.shape(keys)[1], 1])  # ((batch_size, T, embedding_size))

        # 2. 将query，keys，query 与key的差值，元素乘积，链接起来.注意和论文中的外积不一样
        din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)

        # 3. 进入全连接层,输出一个将T 个activation 链接起来的向量
        d_layer_3_all = self.attention_mlp(din_all)  # (batch_size,T,1)
        d_layer_3_all = self.dense_output(d_layer_3_all)
        att_fc_ouputs = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])  # (batch_size,1,T)

        # 4. 有些history 不够长的，用sequence_mask 标记出来
        key_masks = tf.sequence_mask(keys_length,self.max_hist_len, dtype=tf.bool)  # (batch_size,max_length)
        key_masks = tf.expand_dims(key_masks, 1)  # (batch_size,1,max_length)
        paddings = tf.ones_like(att_fc_ouputs) * (-2 ** 32 + 1)  # 在补足的地方附上一个很小的值，而不是0,原因是后面会跟着一个softmax
        # paddings = tf.ones_like(att_fc_ouputs) * (0)
        weights  = tf.where(key_masks, att_fc_ouputs, paddings)

        # 5. Scale
        weights = weights / (keys.get_shape().as_list()[-1] ** 0.5)
        # 是attention计算相似度的公式里需要除以key的长度的平方根


        weights = tf.nn.softmax(weights)
        #TODO: 论文中虽然没有说，但是这里还是采用了softmax的方法，不然之前的padding 很小的值不会起作用。
        # 同时也论证了在不用soft max的情况下，前面不足的。但是似乎收敛变慢了。

        # 6. 输出
        outputs = tf.squeeze(tf.matmul(weights, keys))
        # (batch_size,1,T) *(batch_size,T,embding_size) = (batch_size,1,Embedding_size)
        # squeeze:(batch_size,Embedding_size)
        print("outputs:" + str(outputs.shape))

        return outputs
