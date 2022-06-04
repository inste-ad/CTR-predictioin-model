import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.metrics import AUC
from tensorflow.keras.initializers import RandomNormal, Zeros
from tensorflow.keras.optimizers import Adam

from model.MultitaskModel.MMOE import MMOE, MMOE_one_gate
from model.MultitaskModel.SharedBottom import SharedBottom

column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']
data = pd.read_csv('../../data/cvr/other_sample/census-income.sample', header=None, names=column_names)

data['label_income'] = data['income_50k'].map({' - 50000.': 0, ' 50000+.': 1})
data['label_marital'] = data['marital_stat'].apply(lambda x: 1 if x == ' Never married' else 0)
data.drop(labels=['income_50k', 'marital_stat'], axis=1, inplace=True)

columns = data.columns.values.tolist()
sparse_features = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                   'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                   'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                   'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                   'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                   'vet_question']
dense_features = [col for col in columns if
                  col not in sparse_features and col not in ['label_income', 'label_marital']]

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])



# 记录sparse 的长度

dense_feature_columns = [
    {
        'feat': feat_name,
        'feat_num': 1,
        'embed_dim': None
    }
    for feat_name in dense_features
]
sparse_feature_columns =  [
        {
            'feat': feat_name,
             'feat_num': data[feat_name].max() + 1,
             'embed_dim': 4
         }
        for feat_name in sparse_features
    ]





# 3.generate input data for model

y_1 = data.pop('label_income')
y_2 = data.pop('label_marital')
sparse_X = data[sparse_features]
dense_X =data[dense_features]
dense_X_train, dense_X_test,sparse_X_train, sparse_X_test, y_1_train,y_1_test, y_2_train,y_2_test  = train_test_split(dense_X.values, sparse_X.values,y_1.values,y_2.values, test_size=0.2, random_state=2020)
#

def train_mmoe():
    mmoe_model = MMOE(feature_columns=[dense_feature_columns, sparse_feature_columns], num_experts=3, num_tasks=2,
                      task_types=['binary', 'binary'], expert_dnn_hidden_units=[256, 128],
                      task_ouput_dnn_hidden_units=[64], l2_reg=0.001, use_drop=True,
                      drop_rate=0.5, use_bn=False)
    mmoe_model([dense_X_train[:10], sparse_X_train[:10]])
    mmoe_model.compile(Adam(learning_rate=0.001), loss=["binary_crossentropy", "binary_crossentropy"],
                       metrics=[AUC()])
    history = mmoe_model.fit([dense_X_train, sparse_X_train], [y_1_train, y_2_train],
                             batch_size=256, epochs= 50, verbose=2,
                             validation_data=[[dense_X_test, sparse_X_test], [y_1_test, y_2_test]])
    pred_ans = mmoe_model.predict([dense_X_test, sparse_X_test], batch_size=256)
    print("test income AUC", round(roc_auc_score(y_1_test, pred_ans[0]), 4))
    print("test marital AUC", round(roc_auc_score(y_2_test, pred_ans[1]), 4))

def train_mmoe_one_gate():
    mmoe_ogate_model = MMOE_one_gate(feature_columns=[dense_feature_columns, sparse_feature_columns], num_experts=3, num_tasks=2,
                                     task_types=['binary', 'binary'], expert_dnn_hidden_units=[256, 128],
                                     task_ouput_dnn_hidden_units=[64], l2_reg=0.001, use_drop=True,
                                     drop_rate=0.5, use_bn=False)
    mmoe_ogate_model([dense_X_train[:10], sparse_X_train[:10]])
    mmoe_ogate_model.compile(Adam(learning_rate=0.001), loss=["binary_crossentropy", "binary_crossentropy"],
                             metrics=[AUC()])
    history = mmoe_ogate_model.fit([dense_X_train, sparse_X_train], [y_1_train, y_2_train],
                                   batch_size=256, epochs= 50, verbose=2,
                                   validation_data=[[dense_X_test, sparse_X_test], [y_1_test, y_2_test]])
    pred_ans = mmoe_ogate_model.predict([dense_X_test, sparse_X_test], batch_size=256)
    print("test income AUC", round(roc_auc_score(y_1_test, pred_ans[0]), 4))
    print("test marital AUC", round(roc_auc_score(y_2_test, pred_ans[1]), 4))

def train_share_bottom():
    share_bottom_model = SharedBottom(feature_columns=[dense_feature_columns, sparse_feature_columns], num_experts=3,
                                      num_tasks=2, task_types=['binary', 'binary'], expert_dnn_hidden_units=[256, 128],
                                      task_ouput_dnn_hidden_units=[64], l2_reg=0.001, use_drop=True,
                                      drop_rate=0.5, use_bn=False)
    share_bottom_model([dense_X_train[:10], sparse_X_train[:10]])
    share_bottom_model.compile(Adam(learning_rate=0.001), loss=["binary_crossentropy", "binary_crossentropy"],
                               metrics=[AUC()])
    history = share_bottom_model.fit([dense_X_train, sparse_X_train], [y_1_train, y_2_train],
                                     batch_size=256, epochs= 50, verbose=2,
                                     validation_data=[[dense_X_test, sparse_X_test], [y_1_test, y_2_test]])
    pred_ans = share_bottom_model.predict([dense_X_test, sparse_X_test], batch_size=256)
    print("test income AUC", round(roc_auc_score(y_1_test, pred_ans[0]), 4))
    print("test marital AUC", round(roc_auc_score(y_2_test, pred_ans[1]), 4))


if __name__ == '__main__':
    train_mmoe()

    train_mmoe_one_gate()

    train_share_bottom()

'''
## MMOE (multi gate)
1/1 - 0s - loss: 0.0993 - output_1_loss: 0.0737 - output_2_loss: 0.0169 - output_1_auc: 0.9843 - output_2_auc: 0.9998 - val_loss: 0.9685 - val_output_1_loss: 0.6437 - val_output_2_loss: 0.3160 - val_output_1_auc: 0.7574 - val_output_2_auc: 0.9668 - 79ms/epoch - 79ms/step
test income AUC 0.7843
test marital AUC 0.966
1/1 - 0s - loss: 0.4142 - output_1_loss: 0.1547 - output_2_loss: 0.2512 - output_1_auc: 0.7073 - output_2_auc: 0.9626 - val_loss: 0.6822 - val_output_1_loss: 0.4357 - val_output_2_loss: 0.2381 - val_output_1_auc: 0.8284 - val_output_2_auc: 0.9770 - 83ms/epoch - 83ms/step

'''

'''
'''

'''
## share_bottom_model
1/1 - 0s - loss: 0.1998 - output_1_loss: 0.1174 - output_2_loss: 0.0794 - output_1_auc_1: 0.8810 - output_2_auc_1: 0.9970 - val_loss: 0.6095 - val_output_1_loss: 0.4380 - val_output_2_loss: 0.1683 - val_output_1_auc_1: 0.8750 - val_output_2_auc_1: 0.9872 - 75ms/epoch - 75ms/step
test income AUC 0.8725
test marital AUC 0.9872
2
'''
