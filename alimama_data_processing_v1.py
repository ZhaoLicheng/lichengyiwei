#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 21:08:04 2018

@author: zhaolicheng
"""

#import pandas as pd
#Data = pd.read_csv("/Users/zhaolicheng/Desktop/round1_ijcai_18_train_20180301.txt",\
#                   sep = " ").drop_duplicates().reset_index(drop=True).drop(labels = ["instance_id"],axis = 1)

#Data_test = pd.read_csv("/Users/zhaolicheng/Desktop/round1_ijcai_18_test_a_20180301.txt",\
#                   sep = " ").drop_duplicates()
#Data_test["instance_id"].to_csv("test_instance_id.csv",index = False)
#Data_test.drop(labels = ["instance_id"],axis = 1,inplace = True)
#
#from sklearn import preprocessing
#le = preprocessing.LabelEncoder()
#le.fit(list(Data["item_id"]) + list(Data_test["item_id"]))
#Data["item_id"] = pd.DataFrame(le.transform(Data["item_id"]),columns = ["item_id"])
#Data_test["item_id"] = pd.DataFrame(le.transform(Data_test["item_id"]),columns = ["item_id"])

#le.fit(list(Data["item_brand_id"])+list(Data_test["item_brand_id"]))
#Data["item_brand_id"] = pd.DataFrame(le.transform(Data["item_brand_id"]),columns = ["item_brand_id"])
#Data_test["item_brand_id"] = pd.DataFrame(le.transform(Data_test["item_brand_id"]),columns = ["item_brand_id"])

#le.fit(list(Data["item_city_id"])+list(Data_test["item_city_id"]))
#Data["item_city_id"] = pd.DataFrame(le.transform(Data["item_city_id"]),columns = ["item_city_id"])
#Data_test["item_city_id"] = pd.DataFrame(le.transform(Data_test["item_city_id"]),columns = ["item_city_id"])

#le.fit(list(Data["user_id"])+list(Data_test["user_id"]))
#Data["user_id"] = pd.DataFrame(le.transform(Data["user_id"]),columns = ["user_id"])
#Data_test["user_id"] = pd.DataFrame(le.transform(Data_test["user_id"]),columns = ["user_id"])

#Data.drop(labels = ["context_id"],axis = 1,inplace = True)
#Data_test.drop(labels = ["context_id"],axis = 1,inplace = True)

#le.fit(list(Data["shop_id"])+list(Data_test["shop_id"]))
#Data["shop_id"] = pd.DataFrame(le.transform(Data["shop_id"]),columns = ["shop_id"])
#Data_test["shop_id"] = pd.DataFrame(le.transform(Data_test["shop_id"]),columns = ["shop_id"])

#le.fit(list(Data["item_category_list"]) + list(Data_test["item_category_list"]))
#Data["item_category_list"] = pd.DataFrame(le.transform(Data["item_category_list"]),\
#    columns = ["item_category_list"])
#Data_test["item_category_list"] = pd.DataFrame(le.transform(Data_test["item_category_list"]),\
#    columns = ["item_category_list"])


#Data["user_age_level"] = Data["user_age_level"].map(lambda x: x - 1000 if x >= 1000 else -1)
#Data_test["user_age_level"] = Data_test["user_age_level"].map(lambda x: x - 1000 if x >= 1000 else -1)
#Data["user_occupation_id"] = Data["user_occupation_id"].map(lambda x: x - 2002 if x >= 2002 else -1)
#Data_test["user_occupation_id"] = Data_test["user_occupation_id"].map(lambda x: x - 2002 if x >= 2002 else -1)
#Data["user_star_level"] = Data["user_star_level"].map(lambda x: x - 3000 if x >= 3000 else -1)
#Data_test["user_star_level"] = Data_test["user_star_level"].map(lambda x: x - 3000 if x >= 3000 else -1)
#Data["context_page_id"] = Data["context_page_id"].map(lambda x: x - 4001 if x >= 4001 else -1)
#Data_test["context_page_id"] = Data_test["context_page_id"].map(lambda x: x - 4001 if x >= 4001 else -1)
#Data["shop_star_level"] = Data["shop_star_level"].map(lambda x: x - 4999 if x >= 4999 else -1)
#Data_test["shop_star_level"] = Data_test["shop_star_level"].map(lambda x: x - 4999 if x >= 4999 else -1)


#a,a_test = Data["item_property_list"].copy(),Data_test["item_property_list"].copy()   
#dic_a = {}
#a_list,a_list_test = [],[]
#for i in range(a.shape[0]):
#    tmp = a.iloc[i].split(";")
#    a_list += [tmp]
#    for x in tmp:
#        if dic_a.has_key(x):
#            dic_a[x] += 1
#        else:
#            dic_a[x] = 1
#for i in range(a_test.shape[0]):
#    tmp = a_test.iloc[i].split(";")
#    a_list_test += [tmp]
#    
#cate_important = [x for x in dic_a if dic_a[x]>20000]
#le.fit(cate_important)
#for i in range(len(a_list)):
#    tmp = [x for x in a_list[i] if x in cate_important]
#    tmp = le.transform(tmp)
#    b = ['0']*len(cate_important)
#    for x in tmp:
#        b[x] = '1'
#    a_list[i] = "".join(b)
#for i in range(len(a_list_test)):
#    tmp = [x for x in a_list_test[i] if x in cate_important]
#    tmp = le.transform(tmp)
#    b = ['0']*len(cate_important)
#    for x in tmp:
#        b[x] = '1'
#    a_list_test[i] = "".join(b)
#    
#df_a = pd.DataFrame(a_list,columns = ["item_property_list"])     
#df_a.to_csv("item_property_list/item_property_list.csv",index = False) 
#df_a_test = pd.DataFrame(a_list_test,columns = ["item_property_list"])     
#df_a_test.to_csv("item_property_list/item_property_list_test.csv",index = False) 
#
#a,a_test = Data["predict_category_property"].copy(),Data_test["predict_category_property"].copy()
#dic_b_head = {}
#a_list,a_list_test = [],[]
#for i in range(a.shape[0]):
#    tmp = a.iloc[i].split(";")
#    b_list = []
#    for x in tmp:
#        tmp1 = x.split(":")
#        if len(tmp1) > 1:
#            b_list += [[tmp1[0],tmp1[1].split(",")]]
#        else:
#            b_list += [[tmp1[0]]]
#        if dic_b_head.has_key(tmp1[0]):
#                dic_b_head[tmp1[0]] += 1
#        else:
#            dic_b_head[tmp1[0]] = 1
#    a_list += [b_list]
#for i in range(a_test.shape[0]):
#    tmp = a_test.iloc[i].split(";")
#    b_list = []
#    for x in tmp:
#        tmp1 = x.split(":")
#        if len(tmp1) > 1:
#            b_list += [[tmp1[0],tmp1[1].split(",")]]
#        else:
#            b_list += [[tmp1[0]]]
#    a_list_test += [b_list]
#cate_important = [x for x in dic_b_head if dic_b_head[x]>5000]
#le.fit(cate_important)
#for i in range(len(a_list)):
#    tmp = [x for x in a_list[i] if x[0] in cate_important]   
#    a_list[i] = tmp
#    for x in a_list[i]:
#        x[0] = le.transform([x[0]])[0]
#for i in range(len(a_list_test)):
#    tmp = [x for x in a_list_test[i] if x[0] in cate_important]   
#    a_list_test[i] = tmp
#    for x in a_list_test[i]:
#        x[0] = le.transform([x[0]])[0]
#
#b_list = [[] for _ in range(len(cate_important))]
#for i in range(len(a_list)):
#    for x in a_list[i]:
#        b_list[x[0]] += x[1]
#from collections import Counter
#cate_important_sub = [[] for _ in range(len(cate_important))]
#for i in range(len(cate_important)):
#    count = Counter(b_list[i])
#    cate_important_sub[i] = [x for x in count if count[x]>=100] 
#
#for i in range(len(a_list)):
#    d = ['0'*len(cate_important_sub[k])  for k in range(len(cate_important_sub))]
#    for x in a_list[i]:
#       tmp = [y for y in x[1] if y in cate_important_sub[x[0]]]
#       if len(tmp) > 0 :
#          x[1] = list(le.fit(cate_important_sub[x[0]]).transform(tmp))
#       else:
#          x[1] = []
#       c = ['0']*len(cate_important_sub[x[0]])
#       for z in x[1]:
#           c[z] = '1'
#       x[1] = "".join(c)
#    for x in a_list[i]:
#       d[x[0]] = x[1]
#    a_list[i] = "".join(d) 
#for i in range(len(a_list_test)):
#    d = ['0'*len(cate_important_sub[k])  for k in range(len(cate_important_sub))]
#    for x in a_list_test[i]:
#       tmp = [y for y in x[1] if y in cate_important_sub[x[0]]]
#       if len(tmp) > 0 :
#          x[1] = list(le.fit(cate_important_sub[x[0]]).transform(tmp))
#       else:
#          x[1] = []
#       c = ['0']*len(cate_important_sub[x[0]])
#       for z in x[1]:
#           c[z] = '1'
#       x[1] = "".join(c)
#    for x in a_list_test[i]:
#       d[x[0]] = x[1]
#    a_list_test[i] = "".join(d) 
#
#
#df_a = pd.DataFrame(a_list,columns = ["predict_category_property"])     
#df_a.to_csv("predict_category_property/predict_category_property.csv",index = False) 
#df_a_test = pd.DataFrame(a_list_test,columns = ["predict_category_property"])     
#df_a_test.to_csv("predict_category_property/predict_category_property_test.csv",index = False) 

#Data.drop(labels = ["item_property_list","predict_category_property"],axis = 1,\
#          inplace = True)
#Data.to_csv("Data_rest/Data_rest.csv",index = False)
#Data_test.drop(labels = ["item_property_list","predict_category_property"],axis = 1,\
#          inplace = True)
#Data_test.to_csv("Data_rest/Data_rest_test.csv",index = False)   

#Data["click_hour"] = [x.hour for x in pd.to_datetime(Data["context_timestamp"],unit='s')]
#Data_test["click_hour"] = [x.hour for x in pd.to_datetime(Data_test["context_timestamp"],unit='s')]

#import time
#import datetime
#Data_split = []
#ind_split = []
#for i in range(7):
#    start = (pd.datetime(2018,9,17+i,16,0,0) - pd.datetime(1970, 1, 1)).total_seconds()
#    end = (pd.datetime(2018,9,18+i,15,59,59) - pd.datetime(1970, 1, 1)).total_seconds()
#    ind = (Data["context_timestamp"] >= start) & \
#                 (Data["context_timestamp"] <= end)
#    Data_tmp = Data[ind].drop(labels = \
#                 ["context_timestamp"],axis = 1)
#    Data_tmp.to_csv("alimama/Data_rest/Data_rest_"+str(i+1)+".csv",index = False)
#    ind_split += [ind]
#    Data_split += [Data_tmp]
   
#item_property_list = pd.read_csv("/Users/zhaolicheng/Desktop/alimama/item_property_list/item_property_list.csv") 
#item_property_list_split = []
#for i in range(7):
#    ind = ind_split[i]
#    item_property_list_tmp = item_property_list[ind]
#    item_property_list_tmp.to_csv("alimama/item_property_list/item_property_list_"+str(i+1)+".csv",index = False)
#    item_property_list_split += [item_property_list_tmp]
    
    
#predict_category_property = pd.read_csv("/Users/zhaolicheng/Desktop/alimama/predict_category_property/predict_category_property.csv") 
#predict_category_property_split = []
#for i in range(7):
#    ind = ind_split[i]
#    predict_category_property_tmp = predict_category_property[ind]
#    predict_category_property_tmp.to_csv("alimama/predict_category_property/predict_category_property_"+str(i+1)+".csv",index = False)
#    predict_category_property_split += [predict_category_property_tmp]




import pandas as pd
import numpy as np
import gc
from scipy import sparse
Data_piece_1 = pd.read_csv("/Users/zhaolicheng/Desktop/alimama/output/predict_category_property_1.csv", dtype = np.uint8).to_sparse(fill_value = 0)
Data_piece_1 = sparse.csr_matrix(Data_piece_1.values)
Data_piece_2 = pd.read_csv("/Users/zhaolicheng/Desktop/alimama/output/item_property_list_1.csv", dtype = np.uint8).to_sparse(fill_value = 0)
Data_piece_2 = sparse.csr_matrix(Data_piece_2.values)
Data_piece_3 = pd.read_csv("/Users/zhaolicheng/Desktop/alimama/Data_rest/Data_rest_1.csv")

#%%
from sklearn.preprocessing import OneHotEncoder

def one_hot(df,column_name):
    enc = OneHotEncoder()
    df_full = sorted(pd.DataFrame(list(pd.read_csv("/Users/zhaolicheng/Desktop/alimama/Data_rest/Data_rest.csv")[column_name]) + \
                 list(pd.read_csv("/Users/zhaolicheng/Desktop/alimama/Data_rest/Data_rest_test.csv")[column_name])).drop_duplicates().values.ravel())
    if df_full[0] <= 0:
        minval = df_full[0]
        df_full = [x - df_full[0] for x in df_full]
        enc.fit(np.array(df_full).reshape(-1,1))
        df_id = df[column_name] - minval
        df_id_onehot = enc.transform(df_id.values.reshape(-1,1))
    else:
        enc.fit(np.array(df_full).reshape(-1,1))
        df_id = df[column_name]
        df_id_onehot = enc.transform(df_id.values.reshape(-1,1))
    return df_id_onehot


Data_item_id_onehot = one_hot(Data_piece_3,"item_id")
Data_item_category_list_one_hot = one_hot(Data_piece_3,"item_category_list")   
Data_item_brand_id_onehot = one_hot(Data_piece_3,"item_brand_id")      
Data_item_city_id_onehot = one_hot(Data_piece_3,"item_city_id")     
Data_item_user_id_onehot = one_hot(Data_piece_3,"user_id")
Data_item_user_gender_id_onehot = one_hot(Data_piece_3,"user_gender_id") 
Data_item_user_occupation_id_onehot = one_hot(Data_piece_3,"user_occupation_id") 
Data_item_context_page_id_onehot = one_hot(Data_piece_3,"context_page_id") 
Data_item_shop_id_onehot = one_hot(Data_piece_3,"shop_id") 

y = Data_piece_3["is_trade"]
Data_piece_3.drop(labels = ["item_id","item_category_list","item_brand_id","item_city_id","user_id",\
                            "user_gender_id","user_occupation_id","context_page_id","shop_id","is_trade"],axis = 1,inplace = True)
Data_piece_3 = sparse.csr_matrix(Data_piece_3.values)

from scipy.sparse import hstack
Data_piece_4 = hstack((Data_item_id_onehot,Data_item_category_list_one_hot,\
                       Data_item_brand_id_onehot,Data_item_city_id_onehot,\
                       Data_item_user_id_onehot,Data_item_user_gender_id_onehot,\
                       Data_item_user_occupation_id_onehot,Data_item_context_page_id_onehot,\
                       Data_item_shop_id_onehot))

X = hstack((Data_piece_3,Data_piece_4,Data_piece_1,Data_piece_2))

#%%
from sklearn.metrics import log_loss

import xgboost as xgb
xgbc = xgb.XGBClassifier()
xgbc.fit(X,y)
y_pred = xgbc.predict_proba(X) 
print(log_loss(y,y_pred))

import lightgbm as lgb
lgbc = lgb.LGBMClassifier()
lgbc.fit(X,y)
y_pred = lgbc.predict_proba(X) 
print(log_loss(y,y_pred))

#%% not support
#from neupy.algorithms import PNN
#pnn_network = PNN(std=0.1, verbose=False)
#pnn_network.fit(X, y)
#y_pred = pnn_network.predict_proba(X)
#print(log_loss(y,y_pred))

#from sklearn.ensemble import GradientBoostingClassifier
#gbc = GradientBoostingClassifier()
#gbc.fit(X,y)
#y_pred = gbc.predict_proba(X) 
#print(log_loss(y,y_pred))
#%%  trash code


#for i in range(Data.shape[0]):
#    tmp = sorted(Data.iloc[i]["item_property_list"].split(";"))
#    Data.set_value(i,"item_property_list",";".join(tmp))
#    
#
#le.fit(Data["item_property_list"])
#Data["item_property_list"] = pd.DataFrame(le.transform(Data["item_property_list"]),\
#    columns = ["item_property_list"])



#
#from sklearn.preprocessing import MinMaxScaler
#scalar = MinMaxScaler(feature_range = (0.0,1.0))
#Data["context_timestamp"] = pd.DataFrame(scalar.fit_transform(Data["context_timestamp"].values.reshape(-1,1)),\
#    columns = ["context_timestamp"])
#
#
#a = Data["item_property_list"].copy()  
#dic = {}
#j = 0
#for i in range(a.shape[0]):
#    for x in a.iloc[i].split(";"):
#        if x not in dic:
#            dic[x] = j
#            j += 1
#count = [0]*61407
#for i in range(a.shape[0]):
#    tmp = list(set(a.iloc[i].split(";")))
#    tmp1 = sorted([dic[x] for x in tmp])
#    for x in tmp1:
#       count[x] += 1
#thres = 20000
#Len = len([x for x in count if x > thres])
#for i in range(a.shape[0]):
#    tmp = list(set(a.iloc[i].split(";")))
#    tmp1 = sorted([dic[x] for x in tmp])
#    tmp2 = [x for x in tmp1 if count[x]>thres]
#    a.set_value(i,";".join([str(x) for x in tmp2]))
#b = a.copy()
#dic2 = {}
#j = 0
#for i in range(a.shape[0]):
#    for x in a.iloc[i].split(";"):
#        if x not in dic2:
#            dic2[x] = j
#            j += 1
#for i in range(b.shape[0]):
#    tmp = list(set(b.iloc[i].split(";")))
#    c = ['0']*len(dic2)
#    for x in tmp:
#        c[dic2[x]] = '1'
#    b.set_value(i,";".join(c))
    
#Data["item_property_list"] = b   
#Data.to_csv("train.csv",index = False)          

#s = []
#a = Data["predict_category_property"].copy()
#for i in range(a.shape[0]):
#    for x in a.iloc[i].split(";"):
#        s += [x.split(":")[0]]

#from collections import Counter
#dic = Counter(s)

#cate_important = [x for x in dic if dic[x]>5000]
#print len(cate_important)

#for i in range(a.shape[0]):
#    tmp = a.iloc[i].split(";")
#    tmp = [x for x in tmp if x.split(":")[0] in cate_important]
#    a.set_value(i,";".join(tmp))


#a = [1,3,5,8]
#target = 2
#a = [2,6,7,10]
#target = 0
#a = [1,8,10,12]
#target = 14
#def insert(a,target):
#       left,right = 0,len(a)
#       while left < right:
#          mid = left + (right - left)/2
#          if a[mid] < target:
#              left = mid + 1
#          elif a[mid] > target:
#              right = mid 
#          else:
#              return mid
#       return left
#print insert(a,target)