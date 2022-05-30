import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from model.utils import CustomCallback
from model.DIN import DIN


# ============================Build Model==========================
# din_model = DIN([],[],)
# model = DIN(feature_columns, hist_feature_columns, att_hidden_units, ffn_hidden_units, att_activation,
#     ffn_activation, maxlen, dnn_dropout)
# model.summary()


def test_model():
    dense_features = [{'feat': 'a'}, {'feat': 'b'}]
    sparse_features = [{'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'cate_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'adv_id', 'feat_num': 100, 'embed_dim': 8}]
    hist_feature_columns = ['item_id', 'cate_id']
    features = [dense_features, sparse_features]
    model = DIN(feature_columns, hist_feature_columns)
    model(train_X[:1000])
    model.summary()


if __name__ == '__main__':
    EPOCHS = 50
    BATCH_SIZE = 5000
    USE_DROP = True
    DROPOUT_RATE = 0.5
    LEARNING_RATE = 0.001
    #
    file = '/Users/yangsu/Documents/代码/Myctr/data/din/dataset.pkl'
    #
    with open(file, 'rb') as f:
        feature_columns = pickle.load(f)
        hist_feature_columns = pickle.load(f)
        train = pickle.load(f)
        test = pickle.load(f)

    # train_X 格式：DataFrame Index(['user_id', 'hist', 'target_item', 'cate_ad', 'hist_len'], dtype='object')
    # y 格式：DataFrame  click_label [0,1]
    train_X, train_y = train
    test_X, test_y = test
    dense_feature_columns, sparse_feature_columns  = feature_columns
    hist_feature_columns  = hist_feature_columns




    model = DIN(feature_columns, hist_feature_columns, hist_max_len=20, embed_dim=8, l2_reg=0.01,
                att_units=[80, 40], final_units=[256, 64],
                use_drop=USE_DROP, drop_rate=DROPOUT_RATE, use_bn=False, embed_reg=0.01)

    mini_batch_x  = [x[:10] for x in train_X]
    model(mini_batch_x)


    checkpoint_dir = '../model_weight/DIN'
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

    tensorboard_callback = tf.keras.callbacks.TensorBoard(tensorboard_log_dir, update_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq='epoch',
        period=5
    )

    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=LEARNING_RATE),
                  metrics=[AUC()])

    model.fit(
        train_X,
        train_y,
        epochs=EPOCHS,
        # callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        validation_data=(test_X, test_y),
        # validation_split= 0.2,
        batch_size=BATCH_SIZE,
        callbacks = [CustomCallback(),
                   tensorboard_callback,
                   early_stop,
                   cp_callback
                   ]
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=5000))
