import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
#%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import PowerTransformer

train=pd.read_csv("train.csv", parse_dates=["Scheduled Date","Delivery Date"])
test=pd.read_csv("test.csv", parse_dates=["Scheduled Date","Delivery Date"])
test.sample()

test_id=test["Customer Id"]

train.sample(5)

data=pd.concat((train, test), axis=0)

data.info()

data.nunique()

data.drop(["Customer Id","Artist Name"], axis=1, inplace=True)

data["location"]=data["Customer Location"].str[-8:-6]

data["days"]=(data["Scheduled Date"]- data["Delivery Date"]).dt.days

data.info()

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

cat_col=[]
for i in data.columns:
    if data[i].dtype=='object':
        cat_col.append(i)

for column in cat_col:
    le=LabelEncoder()
    data[column]=le.fit_transform(data[column])

data.head()

data.columns

feat=[x for x in data.columns if x not in ["Cost"]]
x=data[feat]
y=data["Cost"]
x.isnull().sum()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

ran = RandomForestRegressor()
xgb = XGBRegressor()
grad = GradientBoostingRegressor()
lgb = LGBMRegressor()
cat = CatBoostRegressor()

models=[xgb, ran, lgb, grad, cat]

for model in models:
    reg = model
    reg.fit(x_train,np.log(y_train))
    msle = mean_squared_log_error(y_test, np.exp(reg.predict(x_test)))
    print(model.__class__.__name__,":",msle)

reg1 = XGBRegressor()
reg1.fit(x_train,np.log(y_train))
ypred_xgb = pd.Series(reg1.predict(x_test)).abs()

reg = LGBMRegressor()
reg.fit(x_train,np.log(y_train))
ypred_lgb = pd.Series(reg.predict(x_test)).abs()

bst = CatBoostRegressor()
bst.fit(x_train,np.log(y_train))
ypred_cat = pd.Series(bst.predict(x_test)).abs()

bst.fit(x,np.log(y), verbose=200)
pred=pd.DataFrame(prediction,columns=["Cost"])
submission=pd.concat((test_id,pred), axis=1)
submission.head()

submission.to_csv("submission.csv", index=False)
