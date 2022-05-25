#%%
from tensorflow.keras import Model
from model.layers import MLP, LogistRegression,SparseInput,Concatenate,Dense


class DWM(Model):
    def __init__(self, dense_feats, sparse_feats,sparse_dims , embed_dim = 8,hide_units = [256, 128, 64], l2_reg = 0.01, use_drop = True, drop_rate = 0.5, use_bn = False):
        '''

        :param dense_feats: 
        :param sparse_feats:
        :param embed_dim:
        :param hide_units:
        :param l2_reg:
        :param use_drop:
        :param drop_rate:
        :param use_bn:
        '''
        super().__init__()
        self.dense_feats = dense_feats
        self.sparse_feats = sparse_feats
        self.embed_dim =embed_dim
        self.sparse_dims = sparse_dims
        self.deep  = MLP(hide_units,l2_reg,use_drop,drop_rate,use_bn)
        self.wide = LogistRegression()
        self.sparse_input_layer = SparseInput(sparse_feats, sparse_dims, embed_dim, l2_reg)
        self.Concatenate1 = Concatenate(name='dnn_input')
        self.Concatenate2 = Concatenate(name='add_output')
        self.sigmoid_output = Dense(1, activation ='sigmoid')
    def call(self,inputs):
        # dense 在前,saprse 在后
        dense_inputs = inputs[:,:len(self.dense_feats)]
        sparse_inputs = inputs[:,len(self.dense_feats):]
        embed_inputs  = self.sparse_input_layer(sparse_inputs)

        dnn_input =self.Concatenate1 ([dense_inputs,embed_inputs])
        dnn_output = self.deep(dnn_input)

        wide_input =embed_inputs
        wide_output = self.wide(wide_input)

        output  =  self.Concatenate2([dnn_output,wide_output])
        output =   self.sigmoid_output(output)
        return output











