#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
df=pd.read_excel("Data_For Model 20241105.xlsx")


# In[ ]:


import json
import numpy as np

from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from tensorflow.keras.layers import Lambda, Dense,Bidirectional,LSTM
from tensorflow.keras.losses import kullback_leibler_divergence as kld
from tqdm import tqdm
import os
labels = [0,1]


# In[2]:



df["Label"].value_counts()


# In[3]:


tmpd={"IS":0,"ES":1}


# In[4]:


df["label"]=df["Label"].map(tmpd)


# In[5]:


df=df.dropna(subset=["label"])


# In[6]:


df["label"].value_counts()


# In[7]:


df["text"]=df["Text"]
df=df.dropna(subset=["text","label"])


# In[8]:


df


# #### 去除停用词

# In[9]:


import re
import jieba_fast as jieba
ting=[i.strip() for i in open("stop_words_ch-停用词表.txt",encoding='gbk').readlines()]
def fenci(text):
    return [i for i in jieba.cut(str(text)) if i not in ting] 

df["text"]=df["text"].apply(lambda x:fenci(x))


# In[10]:


df["text"]=df["text"].apply(lambda x:" ".join(x))


# ### 加载数据

# In[ ]:


import pandas as pd
num_classes = len(labels)
maxlen = 64

# BERT base
config_path = 'chinese_roberta_www_ext/bert_config.json'
checkpoint_path = 'chinese_roberta_www_ext/bert_model.ckpt'
dict_path = 'chinese_roberta_www_ext/vocab.txt'

from tqdm import tqdm


# In[ ]:


def load_data(df):
    """加载数据
    单条格式：(文本, 标签id)
    """

    D = []
    for i,row in tqdm(df.iterrows()):
        D.append((row["text"], labels.index(row["label"])))
    return D

df=df.dropna(subset=["text"])
df_group=df["id"]
# 加载数据集
df=load_data(df)


# ### 数据加载器

# In[ ]:


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
#             print(text,text1,label)
            token_ids, segment_ids = tokenizer.encode(text, maxlen=50)
            for i in range(1):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([int(label)])
            if len(batch_token_ids) == self.batch_size  or is_end:
                batch_token_ids = sequence_padding(batch_token_ids,length=50)
                batch_segment_ids = sequence_padding(batch_segment_ids,length=50)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# ### 模型搭建

# In[ ]:





# In[ ]:



def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(test_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('bertlstm.weights'.format(1))
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )



# ### 交叉验证

# In[13]:


from sklearn.model_selection import KFold, GroupKFold, ParameterGrid
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
n_splits_inner=5
n_splits_outer=5
inner_kf = KFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
outer_kf = GroupKFold(n_splits=n_splits_outer)
param_grid = {
    'batch_size': [24,64,128],
    'lr': [0.00001,0.00002,0.0001,0.001,0.01],
    'hidden_dim': [32,64,128],
}
param_grid = list(ParameterGrid(param_grid))
for params in param_grid:
    print(params)

params={'batch_size':24,'lr':0.00002,'hidden_dim':128}

# ### 每一折的交叉验证报告 手动平均即为交叉验证的平均结果

# In[14]:


X=np.array(df)

true_labels_all = []
predicted_labels_all = []
for train_index, test_index in outer_kf.split(X,groups=df_group.values.tolist()):


    train_data,test_data=X[train_index],X[test_index]
    train_generator = data_generator(train_data, params['batch_size'])
    test_generator = data_generator(test_data, params['batch_size'])

    for params in param_grid:
        scores=[]
        for inner_train_idx, inner_test_idx in inner_kf.split(train_data):
            train_inner, test_inner = train_data[inner_train_idx], train_data[inner_test_idx]
            train_generator = data_generator(train_inner, params['batch_size'])
            test_generator = data_generator(test_inner, params['batch_size'])
            bert = build_transformer_model(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                return_keras_model=False,
            )
            tmp=bert.model.output
            # tmp=bert.model.output


            # In[ ]:


            # lstmout = Bidirectional(LSTM(128, return_sequences=False))(tmp)
            lstmout = Bidirectional(LSTM(params['hidden_dim'], return_sequences=False))(tmp)
            output = Dense(
                units=len(labels),
                activation='softmax',
                kernel_initializer=bert.initializer
            )(lstmout)
            model = keras.models.Model(bert.model.input, output)
            model.compile(
                loss='sparse_categorical_crossentropy',
                # optimizer=Adam(1e-5),  # 用足够小的学习率
                optimizer=Adam(params['lr']),
                metrics=['sparse_categorical_accuracy']
            )

            evaluator = Evaluator()
            history=model.fit(
                train_generator.forfit(),
                steps_per_epoch=len(train_generator),
                epochs=5, callbacks=[evaluator]
            )

            from tqdm import tqdm
            preall=[]
            trueall=[]
            for x_true, y_true in tqdm(test_generator):
                y_pred = model.predict(x_true).argmax(axis=1)
                preall.append(y_pred)
                trueall.append(np.squeeze(y_true,axis=1))
        print(params, sum(preall == trueall) / len(trueall))
    test_generator = data_generator(test_data, params['batch_size'])
    for x_true, y_true in tqdm(test_generator):
        y_pred = model.predict(x_true).argmax(axis=1)
        preall.append(y_pred)
        trueall.append(np.squeeze(y_true, axis=1))
    preall = np.concatenate(preall, axis=0)
    trueall = np.concatenate(trueall, axis=0)
    true_labels_all.extend(trueall.flatten())
    predicted_labels_all.extend(preall)
    print(params, classification_report(trueall, preall, digits=4, target_names=["IS", "ES"]))
    '''
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False,
    )
    tmp = bert.model.output
    # tmp=bert.model.output

    # In[ ]:

    # lstmout = Bidirectional(LSTM(128, return_sequences=False))(tmp)
    lstmout = Bidirectional(LSTM(params['hidden_dim'], return_sequences=False))(tmp)
    output = Dense(
        units=len(labels),
        activation='softmax',
        kernel_initializer=bert.initializer
    )(lstmout)
    model = keras.models.Model(bert.model.input, output)
    model.compile(
        loss='sparse_categorical_crossentropy',
        # optimizer=Adam(1e-5),  # 用足够小的学习率
        optimizer=Adam(params['lr']),
        metrics=['sparse_categorical_accuracy']
    )

    evaluator = Evaluator()
    history = model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=5, callbacks=[evaluator]
    )

    from tqdm import tqdm

    preall = []
    trueall = []
    for x_true, y_true in tqdm(test_generator):
        y_pred = model.predict(x_true).argmax(axis=1)
        preall.append(y_pred)
        trueall.append(np.squeeze(y_true, axis=1))
    preall = np.concatenate(preall, axis=0)
    trueall = np.concatenate(trueall, axis=0)
    true_labels_all.extend(trueall.flatten())
    predicted_labels_all.extend(preall)
    print(params, classification_report(trueall, preall, digits=4, target_names=["IS", "ES"]))
    '''
# 将结果转换为numpy数组
true_labels_all = np.array(true_labels_all)
predicted_labels_all = np.array(predicted_labels_all)

# 创建一个DataFrame
results_df = pd.DataFrame({
    'True_Label': true_labels_all,
    'Predicted_Label': predicted_labels_all
})

# 将DataFrame保存为CSV文件
results_df.to_csv('bert_lstm_cross_validation.csv', index=False)

