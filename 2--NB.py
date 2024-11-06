#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df=pd.read_excel("Data_For Model 20241105.xlsx")


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


# ### tfidf

# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()  
vectors=vectorizer.fit_transform(df["text"])#训练数据集并且得到tfidf分数


# In[12]:


from sklearn.feature_selection import SelectKBest,f_classif 
selector=SelectKBest(score_func=f_classif,k=5000)#假设性检验  得到和标签最相关的2000个类别
selector.fit(vectors,df["label"])
traindata=selector.transform(vectors).todense()


# ### 交叉验证

# In[13]:


from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
n_splits_outer=5
outer_kf = GroupKFold(n_splits=n_splits_outer)
nb=GaussianNB()
X=traindata
y=df["label"].values
d={"IS":0,"ES":1}
reports=[]


# ### 每一折的交叉验证报告 手动平均即为交叉验证的平均结果

# In[14]:

true_labels_all = []
predicted_labels_all = []
for train_index, test_index in outer_kf.split(X,y,groups=df["id"].values.tolist()):

    # 划分训练集和测试集
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 训练模型
    nb.fit(X_train, y_train)

    # 进行预测
    y_pred_test = nb.predict(X_test)

    # 生成分类报告
    report = classification_report(y_test, y_pred_test,digits=4,target_names=list(d.keys()))
    reports.append(report)
    print(f"Classification report for fold:\n{report}\n")
    true_labels_all.extend(y_test)
    predicted_labels_all.extend(y_pred_test)

for report in reports:
    print(report)


# ### 获得交叉验证的平均结果(更常用)

# In[18]:

# 将结果转换为numpy数组
true_labels_all = np.array(true_labels_all)
predicted_labels_all = np.array(predicted_labels_all)

# 创建一个DataFrame
results_df = pd.DataFrame({
    'True_Label': true_labels_all,
    'Predicted_Label': predicted_labels_all
})

# 将DataFrame保存为CSV文件
results_df.to_csv('nb_cross_validation.csv', index=False)
y_pred = cross_val_predict(nb, X, y, cv=outer_kf.split(X,y,groups=df["id"].values.tolist()))
# 生成总体的分类报告
print(classification_report(y, y_pred,digits=4,target_names=list(d.keys())))

