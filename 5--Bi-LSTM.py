#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
df=pd.read_excel("Data_For Model 20241105.xlsx")


# In[ ]:


from tqdm import tqdm
from tensorflow.keras.layers import *
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedKFold, GroupKFold, ParameterGrid
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
tqdm.pandas()
import os

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


# ### 深度学习模型

# #### w2v加载

# In[ ]:


import pandas as pd
import gensim
if not os.path.exists("w2v"):
    w2v_model = gensim.models.Word2Vec(list(df["text"]), vector_size=128, epochs=10, min_count=0)
    word_vectors = w2v_model.wv
    w2v_model.save("w2v")
else:
    print ("直接加载训练好的w2v模型")
    w2v_model=gensim.models.Word2Vec.load("w2v")
    word_vectors = w2v_model.wv
    print ("w2v模型加载完毕")


# #### 转向量

# In[ ]:


import joblib
x_train=list(df["text"])
if not os.path.exists("tokenizer.joblib"):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)  #统计每个词对应的数字，以便于将文本转化成向量
    joblib.dump(tokenizer,"tokenizer.joblib")
else:
    print("加载token")
    tokenizer=joblib.load("tokenizer.joblib")
train_sequence = tokenizer.texts_to_sequences(x_train)#将所有的文本转化成向量
MAX_SEQUENCE_LENGTH=64 #最大长度
EMBEDDING_DIM = 128 #向量维度

y_train =df["label"]
y_train = to_categorical(y_train)  #将标签 one-hot
y_train = y_train.astype(np.int32)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
train_pad = pad_sequences(train_sequence, maxlen=MAX_SEQUENCE_LENGTH) #将每条文本按照最大长度补0


# #### 嵌入矩阵

# In[ ]:


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM), dtype=np.float32)
not_in_model = 0
in_model = 0
embedding_max_value = 0
embedding_min_value = 1
not_words = []

for word, i in tqdm(word_index.items()):
    if word in w2v_model.wv.key_to_index:
        in_model += 1
        embedding_matrix[i] = np.array(w2v_model.wv[word])
        embedding_max_value = max(np.max(embedding_matrix[i]), embedding_max_value)
        embedding_min_value = min(np.min(embedding_matrix[i]), embedding_min_value)
    else:
        not_in_model += 1
        not_words.append(word)


# #### 构建二分类模型

# In[ ]:


def get_lstmmodel(class_num=2,hidden_states=[64,64],dropout=0.1,lr=0.001):
    embed = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
				  trainable=True)  #定义一个词嵌入层,将句子转化成对应的向量
    inputs_sentence = Input(shape=(MAX_SEQUENCE_LENGTH,))#设置输入向量维度
    sentence =embed(inputs_sentence)#定义词嵌入层
    context1 = Bidirectional(LSTM(hidden_states[0], return_sequences=True))(sentence)  # 双向lstm层,lstm神经元维度为64
    # 全连接层,全连接层神经元维度为100
    context1 = Bidirectional(LSTM(hidden_states[1], return_sequences=False))(context1)
    x=Dropout(dropout)(context1 )
    x = Dense(100)(x)
    output = Dense(class_num, activation='softmax')(x)#softmax层
    model = Model(inputs=[inputs_sentence], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['acc'])#定义损失函数，优化器，评分标准
    model.summary()
    return model


# ### 交叉验证

# In[13]:


from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
n_splits_outer=5
n_splits_inner=5
outer_kf = GroupKFold(n_splits=n_splits_outer)
inner_kf = KFold(n_splits=n_splits_inner, shuffle=True, random_state=42)


param_grid = {
    'batch_size': [24,64,128],
    'hidden_states': [[32,64],[64,64],[64,128]],
    'lr': [0.00001,0.0001,0.001,0.01],
    'dropout':[0.1,0.2,0.5]
}
param_grid = list(ParameterGrid(param_grid))
for params in param_grid:
    print(params)

params={"lr":0.001, 'hidden_states': [64,64],'batch_size':64,'dropout':0.1}
# ### 每一折的交叉验证报告 手动平均即为交叉验证的平均结果

# In[14]:


X=train_pad
y=y_train
true_labels_all = []
predicted_labels_all = []
for train_index, test_index in outer_kf.split(X,y,groups=df["id"].values.tolist()):

    # 划分训练集和测试集
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    for params in param_grid:
        scores = []
        for inner_train_idx, inner_test_idx in inner_kf.split(X_train, y_train):
            X_train_inner, X_test_inner = X_train[inner_train_idx], X_train[inner_test_idx]
            y_train_inner, y_test_inner = y_train[inner_train_idx], y_train[inner_test_idx]
            model = get_lstmmodel(hidden_states=params['hidden_states'], lr=params['lr'],dropout=params['dropout'])
            callbacks = [EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10),
                     ModelCheckpoint("LSTM.hdf5", monitor='val_acc',
                                     mode='max', verbose=0, save_best_only=True,save_weights_only=True)]
            #设置模型提前停止,停止的条件是验证集val_acc两轮已经不增加,保存验证集val_acc最大的那个模型,名称为new_lstm.hdf5
            history=model.fit(X_train_inner,y_train_inner, batch_size=params['batch_size'], epochs=20, callbacks=callbacks,validation_data=(X_test_inner,y_test_inner))

            model.load_weights("LSTM.hdf5")
            testpre=np.argmax(model.predict([X_test]),axis=1)
            import matplotlib.pyplot as plt
            val_loss = history.history['val_loss']
            loss = history.history['loss']
            epochs = range(1, len(loss ) + 1)
            plt.title('lstm_Loss')
            plt.plot(epochs, loss, 'red', label='Training loss')
            plt.plot(epochs, val_loss, 'blue', label='Validation loss')
            plt.legend()
            plt.show()
            plt.cla()

            val_loss = history.history['val_acc']
            loss = history.history['acc']
            scores.append(loss[-1])
            epochs = range(1, len(loss ) + 1)
            plt.title('lstm_acc')
            plt.plot(epochs, loss, 'red', label='Training acc')
            plt.plot(epochs, val_loss, 'blue', label='Validation acc')
            plt.legend()
            plt.show()
            from sklearn.metrics import classification_report
            tmpd={"IS":0,"ES":1}
        print(params, sum(scores) / len(scores))
    model.load_weights("LSTM.hdf5")
    testpre = np.argmax(model.predict([X_test]), axis=1)
    print (classification_report(np.argmax(y_test,axis=1),testpre,digits=4,target_names=list(tmpd.keys())))
    true_labels_all.extend(np.argmax(y_test,axis=1))
    predicted_labels_all.extend(testpre)
'''
    model=get_lstmmodel(hidden_states=params['hidden_states'],lr=params['lr'],dropout=params['dropout'])
    callbacks = [EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10),
             ModelCheckpoint("LSTM.hdf5", monitor='val_acc',
                             mode='max', verbose=0, save_best_only=True,save_weights_only=True)]
    #设置模型提前停止,停止的条件是验证集val_acc两轮已经不增加,保存验证集val_acc最大的那个模型,名称为LSTM.hdf5
    history=model.fit(X_train,y_train, batch_size=params['batch_size'], epochs=20, callbacks=callbacks,validation_data=(X_test,y_test))

    model.load_weights("LSTM.hdf5")
    import matplotlib.pyplot as plt
    val_loss = history.history['val_loss']
    loss = history.history['loss']
    epochs = range(1, len(loss ) + 1)
    plt.title('lstm_Loss')
    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.plot(epochs, val_loss, 'blue', label='Validation loss')
    plt.legend()
    plt.show()
    plt.cla()

    val_loss = history.history['val_acc']
    loss = history.history['acc']
    epochs = range(1, len(loss ) + 1)
    plt.title('lstm_acc')
    plt.plot(epochs, loss, 'red', label='Training acc')
    plt.plot(epochs, val_loss, 'blue', label='Validation acc')
    plt.legend()
    plt.show()
    from sklearn.metrics import classification_report
    tmpd={"IS":0,"ES":1}
    model.load_weights("LSTM.hdf5")
    testpre = np.argmax(model.predict([X_test]), axis=1)
    print (params,classification_report(np.argmax(y_test,axis=1),testpre,digits=4,target_names=list(tmpd.keys())))
    true_labels_all.extend(np.argmax(y_test,axis=1))
    predicted_labels_all.extend(testpre)    '''
# 将结果转换为numpy数组
true_labels_all = np.array(true_labels_all)
predicted_labels_all = np.array(predicted_labels_all)

# 创建一个DataFrame
results_df = pd.DataFrame({
    'True_Label': true_labels_all,
    'Predicted_Label': predicted_labels_all
})

# 将DataFrame保存为CSV文件
results_df.to_csv('lstm_cross_validation.csv', index=False)

