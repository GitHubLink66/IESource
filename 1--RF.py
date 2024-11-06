#!/usr/bin/env python
# coding: utf-8

# In[1]:
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


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


from sklearn.model_selection import KFold, GroupKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier


n_splits_outer=5
n_splits_inner=5
outer_kf = GroupKFold(n_splits=n_splits_outer)
inner_kf = KFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
#rm=RandomForestClassifier(n_estimators=50)
X=traindata
y=df["label"].values
d={"IS":0,"ES":1}
reports=[]
params_list = []
score_list = []

true_labels_all = []
predicted_labels_all = []
# ### 每一折的交叉验证报告 手动平均即为交叉验证的平均结果

# In[21]:

param_grid = {
    'n_estimators':[50,100,200,400]
}
print(df.columns)

for train_index, test_index in outer_kf.split(X,y,groups=df["id"].values.tolist()):

    # 划分训练集和测试集
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 训练模型
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        cv=inner_kf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # 获取最佳参数
    best_params = {k.replace('classifier__', ''): v
                   for k, v in grid_search.best_params_.items()}

    # 在测试集上评估
    y_pred = grid_search.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)

    scores = grid_search.cv_results_['mean_test_score']

    # 打印所有得分
    for params, score in zip(grid_search.cv_results_['params'], scores):
        params_list.append(params)
        score_list.append(score)
        print(f"参数: {params}, 平均得分: {score:.4f}")

    print(f"最佳参数: {best_params}")
    print(f"内层交叉验证最佳得分: {grid_search.best_score_:.4f}")
    print(f"测试集得分: {test_score:.4f}")
    
    # 进行预测
    y_pred_test = grid_search.best_estimator_.predict(X_test)
    
    # 生成分类报告
    report = classification_report(y_test, y_pred_test,digits=4,target_names=list(d.keys()))
    reports.append(report)
    print(f"Classification report for fold:\n{report}\n")
    true_labels_all.extend(y_test)
    predicted_labels_all.extend(y_pred_test)

# ### 获得交叉验证的平均结果(更常用)

# In[23]:

true_labels_all = np.array(true_labels_all)
predicted_labels_all = np.array(predicted_labels_all)

# 创建一个DataFrame
results_df = pd.DataFrame({
    'True_Label': true_labels_all,
    'Predicted_Label': predicted_labels_all
})

# 将DataFrame保存为CSV文件
results_df.to_csv('rm_cross_validation.csv', index=False)


