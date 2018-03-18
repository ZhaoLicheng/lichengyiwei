#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 08:40:13 2018

@author: zhaolicheng
"""

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
    Data_click_hour_onehot = one_hot(Data_piece_3,"click_hour") 

    
    if i != "test":
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
    else:
        Data_piece_3.drop(labels = ["item_id","item_category_list","item_brand_id","item_city_id","user_id",\
                                    "user_gender_id","user_occupation_id","context_page_id","shop_id","click_hour"],axis = 1,inplace = True)
        Data_piece_3 = sparse.csr_matrix(Data_piece_3.values)
        
        
        Data_piece_4 = hstack((Data_item_id_onehot,Data_item_category_list_one_hot,\
                               Data_item_brand_id_onehot,Data_item_city_id_onehot,\
                               Data_user_id_onehot,Data_user_gender_id_onehot,\
                               Data_user_occupation_id_onehot,Data_context_page_id_onehot,\
                               Data_shop_id_onehot,Data_click_hour_onehot))
        
        X = hstack((Data_piece_3,Data_piece_4,Data_piece_1,Data_piece_2))
        return X
         

X2 = get_data_piece("test")
#%%
from sklearn.linear_model import LogisticRegression
weight = {0:1,1:1}
lr = LogisticRegression(penalty = 'l1',C = 0.1, warm_start = True,class_weight = weight)
y_pred = [[] for _ in range(7)]

for i in range(1,8):
    X1,y1 = get_data_piece(i)
    lr.fit(X1,y1)
    y_pred[i-1] = lr.predict_proba(X2)[:,1]
    
#%%
y_pred1 = np.array(y_pred).T
y_pred1 = pd.DataFrame(y_pred1)
y_pred1.to_csv("predict_result.csv",index = False)

#%%
y_pred1 = y_pred1.values
#%%
y_pred2 = 1/(1+np.exp(-np.log(y_pred1/(1-y_pred1)).mean(axis = 1)))  
y_pred2 = pd.DataFrame(y_pred2)
#%%
instance_id = pd.read_csv("/Users/zhaolicheng/Desktop/round1_ijcai_18_test_a_20180301.txt",sep = " ")["instance_id"]
predict = pd.concat((instance_id,y_pred2),axis = 1)
predict.columns = ["instance_id","predicted_score"]
#%%
predict.to_csv("predict_20180317.csv",index = False,sep = " ")