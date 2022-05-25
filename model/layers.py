from tensorflow.keras.layers import Layer,Dense, Dropout, BatchNormalization,Embedding,Concatenate
from tensorflow.keras.regularizers import L1,L2

class MLP(Layer):
    def __init__(self, hide_units = [256, 128, 64], l2_reg = 0.01, use_drop = True, drop_rate = 0.5, use_bn = False):
        '''
        :param hide_units:  list, units of MLP
        :param l2_reg: the constraint of L2 regularization, default is 0, mean no use
        :param use_drop: bool, default is True
        :param drop_rate: flot, [0,1], default is 0.5
        :param use_bn: bool, default is False
        return: 输出network的输出，输出shape [64] units[-1]
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






        


def create_embedding_matrix(feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True):
    from . import feature_column as fc_lib

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict
