import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.regularizers import L2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.layers import MLP, SparseInput


class ESMM(Model):
    def __init__(self, dense_feats, sparse_feats, sparse_dims, embed_dim=8,
                 hide_units=[256, 128, 64], l2_reg=0.01, use_drop=True, drop_rate=0.5, use_bn=False):
        self.dense_feats = dense_feats
        self.sparse_feats = sparse_feats
        self.sparse_dims = sparse_dims
        self.add_layer = tf.keras.layers.Add()

        self.cvr_mlp = MLP(hide_units=hide_units, l2_reg=l2_reg, use_drop=use_drop, drop_rate=drop_rate, use_bn=use_bn)
        self.ctr_mlp = MLP(hide_units=hide_units, l2_reg=l2_reg, use_drop=use_drop, drop_rate=drop_rate, use_bn=use_bn)
        self.share_embedding_layers = []
        for feat in self.sparse_feats:
            self.share_embedding_layers.append(
                Embedding(
                    input_dim=sparse_dims[feat],
                    output_dim=embed_dim,
                    embeddings_regularizer=L2(l2_reg),
                    name=f'Embedding{feat}' )
            )

        self.ctr_output_layer = Dense(1,activation= 'sigmoid')
        self.cvr_output_layer = Dense(1,activation= 'sigmoid')


    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs
        for i in range(len(self.sparse_feats)):
            sparse_emb =  self.share_embedding_layers[i](sparse_inputs[:, i])
        sparse_emb = tf.add_layer(sparse_emb)
        full_inputs  = tf.concatenate([sparse_emb,dense_inputs], axis= -1)

        ctr_mlp_output = self.ctr_mlp(full_inputs)
        ctr = self.ctr_output_layer(ctr_mlp_output)

        cvr_mlp_output = self.cvr_mlp(full_inputs)
        cvr  = self.cvr_output_layer(cvr_mlp_output)
        ctcvr = ctr*cvr
        return [ctr, ctcvr]


