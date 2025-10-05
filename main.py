import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import PowerTransformer

train=pd.read_csv("train.csv", parse_dates=["Scheduled Date","Delivery Date"])
test=pd.read_csv("test.csv", parse_dates=["Scheduled Date","Delivery Date"])
train.shape, test.shape

test.sample()

test_id=test["Customer Id"]

train.sample(5)

data=pd.concat((train, test), axis=0)

data.info()

data.nunique()

data.drop(["Customer Id","Artist Name"], axis=1, inplace=True)

data["location"]=data["Customer Location"].str[-8:-6]
data.drop(["Customer Location"], axis=1, inplace=True)

data["days"]=(data["Scheduled Date"]- data["Delivery Date"]).dt.days
data.drop(["Scheduled Date","Delivery Date"], axis=1, inplace=True)

data.info()

"""
<class 'pandas.core.frame.DataFrame'>
Int64Index: 10000 entries, 0 to 3499
Data columns (total 17 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   Artist Reputation      9028 non-null   float64
 1   Height                 9506 non-null   float64
 2   Width                  9275 non-null   float64
 3   Weight                 9264 non-null   float64
 4   Material               9236 non-null   object 
 5   Price Of Sculpture     10000 non-null  float64
 6   Base Shipping Price    10000 non-null  float64
 7   International          10000 non-null  object 
 8   Express Shipment       10000 non-null  object 
 9   Installation Included  10000 non-null  object 
 10  Transport              8376 non-null   object 
 11  Fragile                10000 non-null  object 
 12  Customer Information   10000 non-null  object 
 13  Remote Location        9229 non-null   object 
 14  Cost                   6500 non-null   float64
 15  location               10000 non-null  object 
 16  days                   10000 non-null  int64  
dtypes: float64(7), int64(1), object(9)
memory usage: 1.4+ MB

"""

mean=np.mean(data["Artist Reputation"])
data["Artist Reputation"].fillna(mean, inplace=True)

mean=np.mean(data["Height"])
data["Height"].fillna(mean, inplace=True)
 
mean=np.mean(data["Width"])
data["Width"].fillna(mean, inplace=True)

mean=np.mean(data["Weight"])
data["Weight"].fillna(mean, inplace=True)

#most_freq=(data["Material"]).mode()[0]

data["Material"].fillna("New", inplace=True)

most_freq=(data["Transport"]).mode()[0]
data["Transport"].fillna("New", inplace=True)

#most_freq=(data["Transport"]).mode()[0]

data["Remote Location"].fillna("New", inplace=True)

data.info()

'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 10000 entries, 0 to 3499
Data columns (total 17 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   Artist Reputation      10000 non-null  float64
 1   Height                 10000 non-null  float64
 2   Width                  10000 non-null  float64
 3   Weight                 10000 non-null  float64
 4   Material               10000 non-null  object 
 5   Price Of Sculpture     10000 non-null  float64
 6   Base Shipping Price    10000 non-null  float64
 7   International          10000 non-null  object 
 8   Express Shipment       10000 non-null  object 
 9   Installation Included  10000 non-null  object 
 10  Transport              10000 non-null  object 
 11  Fragile                10000 non-null  object 
 12  Customer Information   10000 non-null  object 
 13  Remote Location        10000 non-null  object 
 14  Cost                   6500 non-null   float64
 15  location               10000 non-null  object 
 16  days                   10000 non-null  int64  
dtypes: float64(7), int64(1), object(9)
memory usage: 1.4+ MB
'''

cat_col=[]
le=LabelEncoder()
for i in data.select_dtypes(include="object").columns:
    cat_col.append(i)
    data[i]=le.fit_transform(data[i])
    
cat_col

agg_data={
    'Artist Reputation':['sum','max','min','mean'],
    'Price Of Sculpture':['sum','max','min','mean'],
    'Base Shipping Price':['mean','max']
    
}

df = data.groupby(['Material']).agg(agg_data)

df.columns=['A_' + '_'.join(col).strip() for col in df.columns.values]
df.reset_index(inplace=True)
df.head()
data=data.merge(df,on='Material',how='left')

to_drop=[]
corr_matrix = data.drop(['Cost'],axis=1).corr()
# print(corr_matrix)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper   

to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
print(to_drop)

data.drop(to_drop,inplace=True,axis=1)

agg_data={
    'Artist Reputation':['sum','max','min','mean'],
    'Price Of Sculpture':['sum','max','min','mean'],
    'Base Shipping Price':['mean','max']
    
}

df = data.groupby(['Fragile','Customer Information']).agg(agg_data)

df.columns=['B_' + '_'.join(col).strip() for col in df.columns.values]
df.reset_index(inplace=True)

data=data.merge(df,on=['Fragile','Customer Information'],how='left')
data.head()

to_drop=[]
corr_matrix = data.drop(['Cost'],axis=1).corr()
# print(corr_matrix)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper   

to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
print(to_drop)

data.drop(to_drop,inplace=True,axis=1)

data.info()

"""
<class 'pandas.core.frame.DataFrame'>
Int64Index: 10000 entries, 0 to 9999
Data columns (total 31 columns):
 #   Column                      Non-Null Count  Dtype  
---  ------                      --------------  -----  
 0   Artist Reputation           10000 non-null  float64
 1   Height                      10000 non-null  float64
 2   Width                       10000 non-null  float64
 3   Weight                      10000 non-null  float64
 4   Material                    10000 non-null  int32  
 5   Price Of Sculpture          10000 non-null  float64
 6   Base Shipping Price         10000 non-null  float64
 7   International               10000 non-null  int32  
 8   Express Shipment            10000 non-null  int32  
 9   Installation Included       10000 non-null  int32  
 10  Transport                   10000 non-null  int32  
 11  Fragile                     10000 non-null  int32  
 12  Customer Information        10000 non-null  int32  
 13  Remote Location             10000 non-null  int32  
 14  Cost                        6500 non-null   float64
 15  location                    10000 non-null  int32  
 16  days                        10000 non-null  int64  
 17  A_Artist Reputation_sum     10000 non-null  float64
 18  A_Artist Reputation_max     10000 non-null  float64
 19  A_Artist Reputation_min     10000 non-null  float64
 20  A_Artist Reputation_mean    10000 non-null  float64
 21  A_Price Of Sculpture_sum    10000 non-null  float64
 22  A_Base Shipping Price_mean  10000 non-null  float64
 23  A_Base Shipping Price_max   10000 non-null  float64
 24  B_Artist Reputation_sum     10000 non-null  float64
 25  B_Artist Reputation_max     10000 non-null  float64
 26  B_Artist Reputation_min     10000 non-null  float64
 27  B_Artist Reputation_mean    10000 non-null  float64
 28  B_Price Of Sculpture_min    10000 non-null  float64
 29  B_Base Shipping Price_mean  10000 non-null  float64
 30  B_Base Shipping Price_max   10000 non-null  float64
dtypes: float64(21), int32(9), int64(1)
memory usage: 2.1 MB
"""

train=data[~data["Cost"].isna()]
test=data[data["Cost"].isna()].drop("Cost", axis=1)

train.shape, test.shape

x=train.drop("Cost", axis=1)
y=train[["Cost"]].abs()

cat_features_index = [i for i,col in enumerate(x.columns) if col in cat_col]
cat_features_index

x_train, x_test, y_train , y_test= train_test_split(x,y, test_size=0.3, random_state=122)

def test_accuracy(models):
    for i in models:
        i.fit(x_train,np.log(y_train))
        pred=pd.Series(i.predict(x_test)).abs()
        print("{}:{}".format(i,mean_squared_log_error(y_test, np.exp(pred))))


ran=RandomForestRegressor(n_jobs=-1)
grad=GradientBoostingRegressor()
xgb=XGBRegressor()
lgb=LGBMRegressor()
cat=CatBoostRegressor(verbose=200)

models=[xgb, ran, lgb, grad, cat]
test_accuracy(models)

reg1 = XGBRegressor()
reg1.fit(x_train,np.log(y_train),eval_set=[(x_train,np.log(y_train)),(x_test,np.log(y_test))],verbose=200)

ypred_xgb = pd.Series(reg1.predict(x_test)).abs()
mean_squared_log_error(y_test,np.exp(ypred_xgb))

reg = LGBMRegressor()
reg.fit(x_train,np.log(y_train),eval_set=[(x_train,np.log(y_train)),(x_test, np.log(y_test))],verbose=200)

ypred_lgb = pd.Series(reg.predict(x_test)).abs()
mean_squared_log_error(y_test, np.exp(ypred_lgb))

bst = CatBoostRegressor()
bst.fit(x_train,np.log(y_train),eval_set=[(x_test, np.log(y_test))], early_stopping_rounds=100,verbose=200,cat_features=cat_features_index)

ypred_cat = pd.Series(bst.predict(x_test)).abs()
mean_squared_log_error(y_test, np.exp(ypred_cat))

bst.fit(x,np.log(y), verbose=200)
prediction=pd.Series(np.exp(bst.predict(test))).abs()

pred=pd.DataFrame(prediction,columns=["Cost"])
submission=pd.concat((test_id,pred), axis=1)
submission.head()

submission.to_csv("sm.csv", index=False)

plt.figure(figsize=(12,8))
lgb.fit(x,np.log(y))
a=zip(x.columns,lgb.feature_importances_)
feat_imp=pd.DataFrame(a)
feat_imp.columns=["feat","imp"]
feat_imp=feat_imp.sort_values(by="imp", ascending=False)
sns.barplot(data=feat_imp, x="imp", y="feat");
#feat_imp.plot("feat","imp","barh",figsize=(12,8));

feat=list(feat_imp["feat"][:21])
feat

"""
['Price Of Sculpture',
 'Artist Reputation',
 'Base Shipping Price',
 'Weight',
 'Height',
 'Width',
 'days',
 'location',
 'Transport',
 'Express Shipment',
 'A_Price Of Sculpture_sum',
 'Installation Included',
 'International',
 'A_Artist Reputation_mean',
 'B_Artist Reputation_sum',
 'Remote Location',
 'Material',
 'A_Artist Reputation_sum',
 'A_Base Shipping Price_mean',
 'Customer Information',
 'Fragile']
 """
bst=XGBRegressor(max_depth=12, n_estimators=482, learning_rate=0.1,min_child_weight=15)
bst.fit(x[feat],np.log(y))
prediction=pd.Series(np.exp(bst.predict(test[feat]))).abs()

pred=pd.DataFrame(prediction,columns=["Cost"])
submission=pd.concat((test_id,pred), axis=1)
submission.head()

submission.to_csv("sm.csv", index=False)
