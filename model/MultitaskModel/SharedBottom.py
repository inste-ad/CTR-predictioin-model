import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.regularizers import L2
from model.layers import MLP



class SharedBottom(Model):
    def __init__(self,feature_columns,num_experts = 3, num_tasks = 2, task_types = ['binary','regression'] , expert_dnn_hidden_units = [256,128], task_ouput_dnn_hidden_units=[64], l2_reg = 0.001, use_drop= True,drop_rate =0.5, use_bn = False):
        if num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if num_experts <= 1:
            raise ValueError("num_experts must be greater than 1")
        if len(task_types) != num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")
        for task in task_types:
            if task != 'binary' and task !='regression':
                raise ValueError('Please input task_types with "binary" or "regression" ')
        super(SharedBottom,self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.task_types = task_types

        self.embed_sparse_layers = []
        for feat in self.sparse_feature_columns:
            self.embed_sparse_layers.append(
                Embedding(
                    input_dim= feat['feat_num'],
                    output_dim= feat['embed_dim'],
                    embeddings_initializer='random_uniform',
                    embeddings_regularizer= L2(l2_reg),
                    input_length= 1.
                )
            )
        self.share_MLP = MLP(hide_units= [256,128,64 ],l2_reg= l2_reg,use_drop=use_drop,drop_rate = drop_rate, use_bn= use_bn)

        self.task_layers = []
        for i in range(num_tasks):
            self.task_layers.append(
                MLP(task_ouput_dnn_hidden_units,l2_reg= l2_reg,use_drop=use_drop,drop_rate = drop_rate, use_bn= use_bn )
            )
        self.output_layers = []
        for i, task_type in enumerate(task_types):
            if task_type == 'binary':
                self.output_layers.append(
                    Dense(1,activation='sigmoid',name= f'task_{i}'),
                )
            else:
                self.output_layers.append(
                    Dense(1,activation= None,name= f'task_{i}'),
                )



    def call(self,inputs):
        dense_inputs, sparse_inputs = inputs
        emb_inputs = []
        for i,_ in enumerate(self.sparse_feature_columns):
            emb_inputs.append(
                self.embed_sparse_layers[i](sparse_inputs[:,i])
            )
        full_inputs = tf.concat(emb_inputs,axis = -1) # (batch_size, feat_size)
        full_inputs = tf.concat([full_inputs,dense_inputs],axis = -1)

        share_mlp_out = self.share_MLP(full_inputs)

        final_outputs = []
        for i in range(self.num_tasks):
            task_output = self.task_layers[i]( share_mlp_out)
            output = self.output_layers[i]( task_output )
            final_outputs.append( output )


        return  final_outputs
