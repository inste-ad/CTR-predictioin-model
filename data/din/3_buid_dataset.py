import os
import time
import pandas as  pd
import pickle
import random
import numpy as np
import tensorflow as tf
# from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
random.seed(1234)
np.random.seed(1234)



def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dim dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return: dictionary of sparse features. key:feature name, values:dimension
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def create_amazon_electronic_dataset(file, embed_dim=8, maxlen=20):
    """
    :param file: dataset path
    :param embed_dim: latent factor
    :param maxlen:
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start============')
    with open('/Users/yangsu/Documents/代码/Myctr/data/din/remap.pkl', 'rb') as f:
        reviews_df = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, example_count = pickle.load(f)

    reviews_df = reviews_df
    reviews_df.columns = ['user_id', 'item_id', 'time']

    train_data,  test_data = [], []

    for user_id, hist in reviews_df.groupby('user_id'):
        pos_list = hist['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count - 1) # 随机生成一个不在历史记录里的itemid
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list))]
        hist = []
        for i in range(1, len(pos_list)):
            hist_i = pos_list[:i]
            if i == len(pos_list) - 1:
                test_data.append([user_id,hist_i, pos_list[i], cate_list[pos_list[i]], 1])
                test_data.append([user_id,hist_i, neg_list[i], cate_list[neg_list[i]], 0])
            else:
                train_data.append([user_id, hist_i, pos_list[i], cate_list[pos_list[i]], 1])
                train_data.append([user_id,hist_i, neg_list[i], cate_list[neg_list[i]], 0])
                # [user_id,hist_id, ad_id, category of ad ]
                # cate_list: 以id为索引的category的array

    # feature columns
    feature_columns = [[],
                       [sparseFeature('item_id', item_count, embed_dim),
                        sparseFeature('cate_id', cate_count, embed_dim),
                        sparseFeature('user_id', user_count,embed_dim)
                        ]]  #
    # [[], [{'feat': 'item_id', 'feat_num': 63001, 'embed_dim': 8}, {'feat': 'cate_id', 'feat_num': 801, 'embed_dim': 8}]]


    # behavior
    hist_feature_columns = ['item_id']  # , 'cate_id'

    # shuffle
    random.shuffle(train_data)
    random.shuffle(test_data)

    # create dataframe
    # [user_id,hist_id, ad_id, category of ad ]
    train = pd.DataFrame(train_data, columns=['user_id','hist', 'item_id', 'cate_id','label'])
    test = pd.DataFrame(test_data, columns=['user_id','hist', 'item_id',  'cate_id','label'])
    train['hist_len'] = train['hist'].transform(len)
    test['hist_len'] = train['hist'].transform(len)

    print('==================Padding===================')
    # 第一维度填0 表示没有出现的dense feature

    # dense_inputs, sparse_inputs, seq_inputs, item_inputs = inputs
    dense_feature_columns,  sparse_feature_columns = feature_columns
    dense_feats_col = [col['feat'] for col in dense_feature_columns]
    sparse_feats_col = [col['feat'] for col in sparse_feature_columns
                        if col['feat'] not in hist_feature_columns]
    hist_feats_col = [col['feat'] for col in sparse_feature_columns
                      if col['feat'] in hist_feature_columns]  # attention里的feature 种类


    if len(dense_feats_col )== 0:

        train_X = [np.array([0.] * len(train)),
                   train[sparse_feats_col].values,
                   train[hist_feats_col].values,
                   pad_sequences(train['hist'], maxlen=maxlen,padding='post'),
                   train['hist_len'].values,
                 ]
        train_y = train['label'].values
        # dense_inputs, saprse_inputs, hist_target_inputs, hist_inputs
        test_X = [np.array([0.] * len(test)),
                   test[sparse_feats_col].values,
                   test[hist_feats_col].values,
                   pad_sequences(test['hist'], maxlen=maxlen,padding='post'),
                   test['hist_len'].values,
                 ]
        test_y = test['label'].values
    else:
        assert  1 == 0, 'TODO '
        # train_X = [train[dense_feats_col].values,
        #            train[sparse_feats_col].values,
        #            pad_sequences(train['hist'], maxlen=maxlen),
        #            train['hist_len'].values,
        #          ]
        # test_X = [np.array([0.] * len(test)),
        #            test[['item_id',  'cate_id',]].values,
        #            pad_sequences(test['hist'], maxlen=maxlen),
        #           test['hist_len'].values
        #          ]
        # test_y = test['label'].values


    # if no dense or sparse features, can fill with 0
    # train_X = [train['user_id'],
    #            train['hist'],
    #            train['hist'].transform(len),
    #            np.array(train['target_item'].tolist()),
    #            train['cate_ad'],
    #            ]
    # train_y = train['label'].values
    #
    # test_X = [test['user_id'],
    #           test['hist'],
    #           test['hist'].transform(len),
    #           np.array(test['target_item'].tolist()),
    #           test['cate_ad'],
    #           ]
    # test_y = test['label'].values
    print('============Data Preprocess End=============')
    return feature_columns, hist_feature_columns, (train_X, train_y), (test_X, test_y)


file = '/Users/yangsu/Documents/代码/Myctr/data/din/remap.pkl'
maxlen = 20

embed_dim = 8
att_hidden_units = [80, 40]
ffn_hidden_units = [256, 128, 64]
dnn_dropout = 0.5
att_activation = 'sigmoid'
ffn_activation = 'prelu'

learning_rate = 0.001
batch_size = 4096
epochs = 5
# ========================== Create dataset =======================
# %%
feature_columns, hist_feature_columns, train, test = create_amazon_electronic_dataset(file, embed_dim, maxlen)
# train 的组成：[user_id,hist_id,item_id,label]


file = '/Users/yangsu/Documents/代码/Myctr/data/din/dataset.pkl'

with open(file, 'wb') as f:
    pickle.dump(feature_columns, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(hist_feature_columns, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)  #
    pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)
