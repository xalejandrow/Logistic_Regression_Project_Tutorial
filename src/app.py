# import pandas as pd
# import numpy as np
# import seaborn as sns
# import pickle
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_recall_fscore_support
# from sklearn.model_selection import train_test_split

# #url="https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv"
# url="../data/raw/bank-marketing-campaign-data.csv"
# df=pd.read_csv(url, sep=';')
# #Quito datos duplicados
# df=df.drop_duplicates(subset=columns_names.tolist(), keep='last')
# #Sustituyo unknowns por valor mas frecuente de cada categoria
# df_cat = df.select_dtypes(object)
# cat = df_cat.columns.values.tolist()
# unknown_list=[]
# for i in cat:
#     for z in list(df[i].value_counts().index):
#         if z == 'unknown':
#             unknown_list.append(i)
# value_fr_max={}
# for i in unknown_list:
#     if df[i].value_counts().to_numpy()[0] == df[i].value_counts().max():
#         key = i
#         value = df[i].value_counts().index.to_numpy()[0]
#         value_fr_max[key]=value
# def replace_uknown(b, z):
#     if z == 'unknown': 
#         return value_fr_max[b]
#     else:
#         return z
# for i in unknown_list:
#     df[f"{i}_cor"] = df.apply(lambda x: replace_uknown(i, x[i]), axis=1)
# #elimino outliers
# df = df[(df["age"] < 69.5) & (df["duration"] < 644.5) & (df["campaign"] < 6.0)]
# #convierto age a level numerico
# def age_group(a):
#     if a <= 10: 
#         return '10' 
#     elif a <= 20 and a > 10: 
#         return '20' 
#     elif a <= 30 and a > 20: 
#         return '30'
#     elif a <= 40 and a > 30: 
#         return '40'
#     elif a <= 50 and a > 40: 
#         return '50'
#     elif a <= 60 and a > 50: 
#         return '60'
#     elif a <= 70 and a > 60: 
#         return '70' 
#     elif a <= 80 and a > 70: 
#         return '80'
#     elif a <= 90 and a > 80: 
#         return '90'                        
#     else: return '100'     
# df['age_cor'] = df['age'].apply(age_group)
# #defino categorica midle_school
# def middle_school(z):
#     if z == 'basic.9y': 
#         return 'middle_school'
#     if z == 'basic.6y': 
#         return 'middle_school'
#     if z == 'basic.4y': 
#         return 'middle_school'        
#     else:
#         return z
# df['education_cor_cor'] = df['education_cor'].apply(middle_school) 
# # defino binaria a y
# df_y_dummies = pd.get_dummies(df['y'])
# df_y_dummies = df_y_dummies.drop(['no'], axis=1)
# df_y_dummies=df_y_dummies.rename(columns={'yes': 'y_bin'})
# df = pd.concat((df_y_dummies, df), axis=1)
# # pdays a categorica
# def pdays(z):
#     if z == 999: 
#         return 0  
#     else:
#         return 1
# df['pdays_bin'] = df['pdays'].apply(pdays)
# #level numerico a variables categoricas
# cat_bin =['job_cor', 'marital_cor', 'default_cor', 'housing_cor', 'loan_cor', 'age_cor', 'education_cor_cor', 'contact', 'month', 'day_of_week', 'poutcome']
# cat_bin_d={}
# for i in cat_bin:
#     key = i
#     value = list(df[i].value_counts().index), len(list(df[i].value_counts().index))
#     cat_bin_d[key]=value 
# def replace_bin(b, z):
#     for i in range(cat_bin_d[b][1]):
#         if z == cat_bin_d[b][0][i]:
#             return i                  
# for i in cat_bin:
#     df[f"{i}_bin"] = df.apply(lambda x: replace_bin(i, x[i]), axis=1)
# #variable continuas a normalizar (saco a las categoricas con level numerico y la variable dependiente)
# num_2=['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
# def normalize(b, z):
#     return (z-df[b].min())/(df[b].max()-df[b].min())
# for i in num_2:
#     df[f"{i}_N"] = df.apply(lambda x: normalize(i, x[i]), axis=1)
# #
# df_num_3 = df.select_dtypes(np.number)
# num_3_ = df_num_3.columns.values.tolist()
# X = df[num_3_].drop(['y_bin','age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'], axis=1)
# Y = df[num_3_]['y_bin']
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, train_size = 0.75)
# #modelo
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(X_train, Y_train)
# #Guardo modelo
# pickle.dump(model, open('../models/best_model.pickle', 'wb'))