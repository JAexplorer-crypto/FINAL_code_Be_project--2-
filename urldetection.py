# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 16:41:45 2021

@author: Juhi
"""


import numpy as np
import pandas as pd
import re
import nltk
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')
    allTokens=[]
    for i in tokensBySlash:
        tokens = str(i).split('-')
        tokensByDot = []
        for j in range(0,len(tokens)):
            tempTokens = str(tokens[j]).split('.')
            tokensByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))
    print(allTokens)
    if 'com' in allTokens:
        allTokens.remove('com')
    return allTokens

def detect_malicious_url(test_url):

    data_dir =r"C:\Users\Juhi\Documents\BE project\urldata.csv"
    df = pd.read_csv(data_dir) 
    df = pd.DataFrame(df)
    df = df.sample(n=10000)

    
    col = ['label','url']
    df = df[col]
    #Deleting nulls
    df = df[pd.notnull(df['url'])]
    #more settings for our data manipulation
    df.columns = ['label', 'url']
    df['category_id'] = df['label'].factorize()[0]
    category_id_df = df[['label', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'label']].values)
    # test_url="https://www.youtube.com/watch?v=fwY9Qv96DJY"
    # import matplotlib.pyplot as plt
    # #%matplotlib inline
    # BAD_len = df[df['label'] == 'bad'].shape[0]
    # print(BAD_len)
    # GOOD_len = df[df['label'] == 'good'].shape[0]
    # print(GOOD_len)
    # plt.bar(10,BAD_len,3, label="BAD URL")
    # plt.bar(15,GOOD_len,3, label="GOOD URL")
    # plt.legend()
    # plt.ylabel('Number of examples')
    # plt.title('Propoertion of examples')
    # plt.show()

    # import matplotlib.pyplot as plt
    # #%matplotlib inline
    # lens = df.url.str.len()
    # lens.hist(bins = np.arange(0,300,10))

    y = [d[1]for d in df] #labels
    myUrls = [d[0]for d in df] #urls 

    vectorizer = TfidfVectorizer(tokenizer=getTokens,use_idf=True, smooth_idf=True, sublinear_tf=False)
    features = vectorizer.fit_transform(df.url).toarray()
    labels = df.label
    features.shape
    model = LogisticRegression(random_state=0)
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.20, random_state=0)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    clf = LogisticRegression(random_state=0) 
    clf.fit(X_train,y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print ('train accuracy =', train_score)
    print ('test accuracy =', test_score)

    #conf_mat = confusion_matrix(y_test, y_pred)
    # sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=category_id_df.label.values, yticklabels=category_id_df.label.values)
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')

    X_predict = [test_url]
    X_predict = vectorizer.transform(X_predict)
    #print(X_predict)
    y_Predict = clf.predict(X_predict)[0]
    print(y_Predict)
    return y_Predict