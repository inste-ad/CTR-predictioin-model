import os
import time
import pandas as  pd
import pickle
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.DIN import DIN
 # ============================Build Model==========================
 # din_model = DIN([],[],)
# model = DIN(feature_columns, behavior_list, att_hidden_units, ffn_hidden_units, att_activation,
#     ffn_activation, maxlen, dnn_dropout)
# model.summary()


#
file = '/Users/yangsu/Documents/代码/Myctr/data/din/dataset.pkl'
#
with open(file, 'rb') as f:
    behavior_list = pickle.load( f)
    feature_columns= pickle.load( f)
    train = pickle.load( f)
    test = pickle.load( f)

# train_X 格式：DataFrame Index(['user_id', 'hist', 'target_item', 'cate_ad', 'hist_len'], dtype='object')
# y 格式：DataFrame  click_label [0,1]
train_X, train_y = train
test_X, test_y = test



def test_model():
    # dense_features = [{'feat': 'a'}, {'feat': 'b'}]
    # sparse_features = [{'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8},
    #                    {'feat': 'cate_id', 'feat_num': 100, 'embed_dim': 8},
    #                    {'feat': 'adv_id', 'feat_num': 100, 'embed_dim': 8}]
    # behavior_list = ['item_id', 'cate_id']
    # features = [dense_features, sparse_features]
    model = DIN(feature_columns, behavior_list)
    model(train_X[:1000])
    model.summary()

if __name__ == '__main__':
    test_model()
