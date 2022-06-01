import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.regularizers import L2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.layers import MLP, Concatenate, LocalActivationLayer


class DIN(Model):
    def __init__(self, feature_columns, hist_feature_columns, hist_max_len=20, embed_dim=8, l2_reg=0.01,
                 att_units=[80, 40], final_units=[256, 64],
                 use_drop=True, drop_rate=0.5, use_bn=False, embed_reg=0.01):
        super(DIN, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.max_hist_len = hist_max_len
        self.hist_feature_columns = hist_feature_columns

        # len
        self.sparse_len = len(self.sparse_feature_columns) - len(hist_feature_columns)
        self.dense_len = len(self.dense_feature_columns)
        self.hist_feature_len = len(hist_feature_columns)

        self.embed_dim = embed_dim
        self.use_drop = use_drop
        self.drop_rate = drop_rate
        self.use_bn = use_bn

        # 普通的Cate embeddings
        self.embed_sparse_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=L2(embed_reg),
                                              mask_zero=True
                                              )
                                    for feat in self.sparse_feature_columns
                                    if feat['feat'] not in hist_feature_columns]

        # hitorical  behavior embedding layers, item id and category id
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                           input_length=1,
                                           output_dim=feat['embed_dim'],
                                           embeddings_initializer='random_uniform',
                                           embeddings_regularizer=L2(embed_reg),
                                           mask_zero=True
                                           )
                                 for feat in self.sparse_feature_columns
                                 if feat['feat'] in hist_feature_columns]

        self.attention_layer = LocalActivationLayer(att_units, max_hist_len= self.max_hist_len)
        self.din_mlp = MLP(hide_units=final_units, use_drop=self.use_drop, drop_rate=self.drop_rate, use_bn=False)
        self.sigmoid_output = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform',
                                    kernel_regularizer=L2(l2_reg))


    def call(self, inputs):
        """
        :param inputs:  [dense_inputs, sparse_inputs, sparse_att_inputs,  hist_inputs, hist_length], 其中dense_feats 被全填充为0
        :return: probability
        """
        dense_inputs, sparse_inputs, sparse_att_inputs,  hist_inputs, hist_length = inputs

        # 普通sparse feats
        sparse_emb = []
        for i in range(self.sparse_len):
            sparse_emb.append(self.embed_sparse_layers[i](sparse_inputs[:, i]))
        sparse_emb = tf.concat(sparse_emb, axis=-1)

        # concat dense if exists
        if     self.dense_len  != 0:
            original_emb = tf.concat([dense_inputs, sparse_emb])
        else:
            original_emb = sparse_emb

        # 填充历史数据为等长度
        if self.hist_feature_len == 1:
            query_emb = self.embed_seq_layers[0](sparse_att_inputs[:, 0])
            hist_emb = self.embed_seq_layers[0](hist_inputs)
        else:
            # 非只有一个  TODO:可能不对
            assert 1 == 0, 'wait to do'
            # seq_embed = []
            # query_emb = []
            # for i in range(self.hist_feature_len):
            #     query_emb.append(self.embed_seq_layers[i](sparse_inputs[:, i]))
            #     for j in range(hist_length[i]):
            #         seq_embed.append(self.embed_seq_layers[i](hist_inputs[i, j]))
            # seq_embed = tf.concat(seq_embed)
            query_emb = tf.concat(query_emb)

        # attention
        attention_emb = self.attention_layer([query_emb, hist_emb, hist_length])  # (None, d * 2)
        # concatenate all embeddings
        all_emb =  tf.concat([attention_emb, original_emb], axis = -1)
        mlp_outputs = self.din_mlp(all_emb)
        outputs = self.sigmoid_output(mlp_outputs)

        return outputs



