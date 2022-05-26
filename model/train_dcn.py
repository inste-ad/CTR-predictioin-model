from sklearn.model_selection import train_test_split

from model.DCN import DCN
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import pandas as pd
import  tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.metrics import AUC
from sklearn.metrics import log_loss, roc_auc_score
import os

def generate_sparse_dim_dic(feat, feat_num, embed_dim=4):
    """
    为稀疏特征构建字典
    :@param feat: 特征名称
    :@param feat_num: 不重复的稀疏特征个数
    :@param embed_dim: 特征嵌入(embedding)的维度
    """
    return {'feat_name': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


df = pd.read_csv('../data/train.txt',nrows= 5000000,sep='\t',header=None)

names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13',
             'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
             'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
             'C23', 'C24', 'C25', 'C26']
df.columns = names

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]
features = dense_features+ sparse_features



df[sparse_features] = df[sparse_features].fillna('nan')
df[dense_features] = df[dense_features].fillna(0)

# Dense label
mms = MinMaxScaler(feature_range=(0, 1))
df[dense_features] = mms.fit_transform(df[dense_features])

# sparse transform
for feat in sparse_features:
    lbe = LabelEncoder()
    df[feat] = lbe.fit_transform(df[feat])


sparse_dim_dic = {feat:int(df[feat].max()) + 1 for feat in sparse_features }

#%%
y = df.pop('label')
X = df = df[features] # dense在前，sparse 在后
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2,random_state= 42)
#%%


def train_deep_cross():
    checkpoint_dir = '../model_weight/DCN'
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, '"cp-{epoch:04d}.ckpt"')

    tensorboard_log_dir = os.path.join(checkpoint_dir, "../logs")
    if not os.path.isdir(tensorboard_log_dir):
        os.mkdir(tensorboard_log_dir)

    history_path = os.path.join(checkpoint_dir, "history/hist.csv")
    history_dir = os.path.dirname(history_path)
    if not os.path.isdir(history_dir):
        os.mkdir(history_dir)


    deep_cross_model = DCN(dense_feats=dense_features, sparse_feats=sparse_features, sparse_dims=sparse_dim_dic)
    deep_cross_model(X_train.values[:10])
    deep_cross_model.compile(loss=binary_crossentropy,
                          optimizer=RMSprop(learning_rate=0.005),
                          metrics=[AUC(), binary_crossentropy])

    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 10 == 0:
                print(f'epoch {epoch}')
            print(',\n', end='')

        def on_batch_begin(self, batch, logs):
            if batch % 100 == 0:
                print('.', end='')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(tensorboard_log_dir, update_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq='epoch',
        period=5
    )
    history = deep_cross_model.fit(X_train.values,
                                y_train.values.reshape(-1, 1),
                                batch_size=50000,
                                epochs=200,
                                verbose=2,
                                validation_split=0.2,
                                callbacks=[CustomCallback(),
                                           tensorboard_callback,
                                           early_stop,
                                           cp_callback
                                           ]
                                )
    pred_ans = deep_cross_model.predict(X_test.values, batch_size=256)
    print("test LogLoss", round(log_loss(y_test.values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(y_test.values, pred_ans), 4))

train_deep_cross()