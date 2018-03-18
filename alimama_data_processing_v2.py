#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 21:08:04 2018

@author: zhaolicheng
"""

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from scipy.sparse import hstack


def get_data_piece(i):
    Data_piece_1 = pd.read_csv("/Users/zhaolicheng/Desktop/alimama/output/predict_category_property_"+str(i)+".csv", dtype = np.uint8).to_sparse(fill_value = 0)
    Data_piece_1 = sparse.csr_matrix(Data_piece_1.values)
    Data_piece_2 = pd.read_csv("/Users/zhaolicheng/Desktop/alimama/output/item_property_list_"+str(i)+".csv", dtype = np.uint8).to_sparse(fill_value = 0)
    Data_piece_2 = sparse.csr_matrix(Data_piece_2.values)
    Data_piece_3 = pd.read_csv("/Users/zhaolicheng/Desktop/alimama/Data_rest/Data_rest_"+str(i)+".csv")
    
    
    def one_hot(df,column_name):
        enc = OneHotEncoder()
        if column_name != "click_hour":
            df_full = sorted(pd.DataFrame(list(pd.read_csv("/Users/zhaolicheng/Desktop/alimama/Data_rest/Data_rest.csv")[column_name]) + \
                         list(pd.read_csv("/Users/zhaolicheng/Desktop/alimama/Data_rest/Data_rest_test.csv")[column_name])).drop_duplicates().values.ravel())
        else:
            df_full = range(24)
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
    Data_user_id_onehot = one_hot(Data_piece_3,"user_id")
    Data_user_gender_id_onehot = one_hot(Data_piece_3,"user_gender_id") 
    Data_user_occupation_id_onehot = one_hot(Data_piece_3,"user_occupation_id") 
    Data_context_page_id_onehot = one_hot(Data_piece_3,"context_page_id") 
    Data_shop_id_onehot = one_hot(Data_piece_3,"shop_id") 
    try:
       Data_click_hour_onehot = one_hot(Data_piece_3,"click_hour") 
    except:
       print Data_piece_3.columns
    
    y = Data_piece_3["is_trade"]
    Data_piece_3.drop(labels = ["item_id","item_category_list","item_brand_id","item_city_id","user_id",\
                                "user_gender_id","user_occupation_id","context_page_id","shop_id","is_trade","click_hour"],axis = 1,inplace = True)
    Data_piece_3 = sparse.csr_matrix(Data_piece_3.values)
    
    
    Data_piece_4 = hstack((Data_item_id_onehot,Data_item_category_list_one_hot,\
                           Data_item_brand_id_onehot,Data_item_city_id_onehot,\
                           Data_user_id_onehot,Data_user_gender_id_onehot,\
                           Data_user_occupation_id_onehot,Data_context_page_id_onehot,\
                           Data_shop_id_onehot,Data_click_hour_onehot))
    
    X = hstack((Data_piece_3,Data_piece_4,Data_piece_1,Data_piece_2))
    return X,y

i = 4
X1,y1 = get_data_piece(i)
X2,y2 = get_data_piece(1)

#%%
#from imblearn.under_sampling import RandomUnderSampler
#
##ratio = 'auto'
#ratio = {1:sum(y1),0:10*sum(y1)}
#X1_res, y1_res = RandomUnderSampler(ratio=ratio, random_state=5).fit_sample(X1, y1)

#%%
from sklearn.linear_model import LogisticRegression
weight = {0:1,1:1}
lr = LogisticRegression(penalty = 'l1',C = 0.1, warm_start = True,class_weight = weight)
lr.fit(X1,y1)
#lr = LogisticRegression(penalty = 'l1',C = 1, warm_start = True,class_weight = weight)
#lr.fit(X1_res,y1_res)
#lr.intercept_[0] += np.log(10*sum(y1)*0.7/(len(y1)))
y_pred1 = lr.predict_proba(X1) 
print(log_loss(y1,y_pred1))
a1 = sorted(y_pred1[:,1],reverse = True)
y_pred2 = lr.predict_proba(X2) 
print(log_loss(y2,y_pred2))
a2 = sorted(y_pred2[:,1],reverse = True)
#%%
#import xgboost as xgb
#xgbc = xgb.XGBClassifier()
#xgbc.fit(X,y)
#y_pred = xgbc.predict_proba(X) 
#print(log_loss(y,y_pred))
#%%
#import lightgbm as lgb
#lgbc = lgb.LGBMClassifier()
#lgbc.fit(X,y)
#y_pred = lgbc.predict_proba(X) 
#print(log_loss(y,y_pred))

##%% not support
#from neupy.algorithms import PNN
#pnn_network = PNN(std=0.1, verbose=False)
#pnn_network.fit(X, y)
#y_pred = pnn_network.predict_proba(X)
#print(log_loss(y,y_pred))
#%%
#from sklearn.ensemble import GradientBoostingClassifier
#gbc = GradientBoostingClassifier()
#gbc.fit(X,y)
#y_pred = gbc.predict_proba(X) 
#print(log_loss(y,y_pred))
