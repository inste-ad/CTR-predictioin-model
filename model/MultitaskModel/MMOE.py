import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.regularizers import L2
from model.layers import MLP, SparseInput



class MMOE(Model):
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
        super(MMOE,self).__init__()
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

        self.gates_layers = []
        for i in range(num_tasks):
            self.gates_layers.append(
                Dense(self.num_experts, activation= 'softmax',use_bias= True, bias_regularizer=L2(l2_reg),kernel_regularizer=L2(l2_reg))
            )
        self.expert_layers = []
        for i in range(num_experts):
            self.expert_layers.append(
                MLP(expert_dnn_hidden_units,l2_reg= l2_reg,use_drop=use_drop,drop_rate = drop_rate, use_bn= use_bn)
            )
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
        # expert network
        expert_outputs = []
        for i in range(self.num_experts):
            expert_outputs.append(
                tf.expand_dims(
                    self.expert_layers[i](full_inputs), #  (batch_size, output_units)
                    axis= -1
                )
            ) #  (batch_size, output_units,1)
        expert_outputs = tf.concat(expert_outputs,axis = -1)
        #  (batch_size, output_units, num_experts)

        gate_outputs = []
        for i in range(self.num_tasks):
            gate_outputs.append(
                tf.expand_dims(
                    self.gates_layers[i](full_inputs), #  (batch_size, num_experts)
                    axis= -1
                ) #  (batch_size, num_experts, 1)
            )
        gate_outputs = tf.concat(gate_outputs, axis = -1)
        #  (batch_size, num_experts, num_tasks)

        # task_inputs = tf.tensordot(
        #     expert_outputs,
        #     gate_outputs,
        #     axes = [[1,2],[1,2]]
        # )  # need (batch_size, output_units,num_tasks)

        task_inputs = tf.matmul(
            expert_outputs,
            gate_outputs,
        )  # need (batch_size, output_units,num_tasks)

        final_outputs = []
        for i in range(self.num_tasks):
            task_output = self.task_layers[i]( task_inputs[:, :, i]) # 这里隐式的降维了。 本来需要用 tf.squeeze
            output = self.output_layers[i]( task_output )
            final_outputs.append( output )


        return  final_outputs




class MMOE_one_gate(Model):
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
        super(MMOE_one_gate,self).__init__()
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

        self.gates_layer = Dense(self.num_experts, activation= 'softmax',use_bias= True, bias_regularizer=L2(l2_reg),kernel_regularizer=L2(l2_reg))

        self.expert_layers = []
        for i in range(num_experts):
            self.expert_layers.append(
                MLP(expert_dnn_hidden_units,l2_reg= l2_reg,use_drop=use_drop,drop_rate = drop_rate, use_bn= use_bn)
            )
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
        # expert network
        expert_outputs = []
        for i in range(self.num_experts):
            expert_outputs.append(
                tf.expand_dims(
                    self.expert_layers[i](full_inputs), #  (batch_size, output_units)
                    axis= -1
                )
            ) #  (batch_size, output_units,1)
        expert_outputs = tf.concat(expert_outputs,axis = -1)
        #  (batch_size, output_units, num_experts)

        ''' 区别在这里只有一个gate共享输出'''
        gate_outputs =tf.expand_dims( self.gates_layer(full_inputs),axis= -1) #  (batch_size, num_experts)
        task_inputs = tf.matmul(
            expert_outputs,
            gate_outputs,
        )  # need (batch_size, output_units,num_tasks)
        task_inputs = tf.squeeze(task_inputs)
        ''' 共享task_inputs '''
        final_outputs = []
        for i in range(self.num_tasks):
            task_output = self.task_layers[i]( task_inputs)
            output = self.output_layers[i]( task_output )
            final_outputs.append( output )

        return  final_outputs

