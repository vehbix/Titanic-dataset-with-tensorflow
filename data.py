import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split



test_veriler=pd.read_csv("test.csv")
train_veriler=pd.read_csv("train.csv")

test_DF=pd.DataFrame(test_veriler)
train_DF=pd.DataFrame(train_veriler)

test_DF.drop(labels=["Name","Ticket","Fare","Cabin"],axis=1,inplace=True)
train_DF.drop(labels=["PassengerId","Name","Ticket","Fare","Cabin"],axis=1,inplace=True)
train_DF=train_DF.dropna()
test_DF=test_DF.dropna()

le=LabelEncoder()
train_DF["Sex"]=le.fit_transform(train_DF["Sex"]) 
test_DF["Sex"]=le.fit_transform(test_DF["Sex"])#male 1 female 0 
train_DF["Embarked"]=le.fit_transform(train_DF["Embarked"])
test_DF["Embarked"]=le.fit_transform(test_DF["Embarked"]) 


# sbn.scatterplot(x="Age",y="Survived",data=train_DF)
# plt.show()
# print(train_DF.groupby("Age").sum()["Survived"])

train_DF_99= train_DF.sort_values("Age",ascending = False).iloc[7:]

# print(train_DF_99.groupby("Age").sum()["Survived"])
# sbn.scatterplot(x="Age",y="Survived",data=train_DF_99)
# plt.show()
#0-12 55+


ageScaler=StandardScaler()
train_DF_99["Age"]=ageScaler.fit_transform(train_DF_99[["Age"]])
train_DF["Age"]=ageScaler.fit_transform(train_DF[["Age"]])
test_DF["Age"]=ageScaler.fit_transform(test_DF[["Age"]])



x=train_DF_99.drop(labels=["Survived"],axis=1,inplace=False) #x -> feature(özellik)
y=train_DF_99["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=15)




# deneme_train=train_DF_99.copy()
# a=deneme_train["Age"]
# ageGroupTrain=[]
# for i in a:
#     if i<=12:
#         ageGroupTrain.append(0)
#     if 12<i and i<55:
#         ageGroupTrain.append(1)
#     if i>=55:
#         ageGroupTrain.append(2)
# deneme_train['Age'] = ageGroupTrain


# deneme_test=test_DF.copy()
# a=deneme_test["Age"]
# ageGroupTest=[]
# for i in a:
#     if i<=12:
#         ageGroupTest.append(0)
#     if 12<i and i<55:
#         ageGroupTest.append(1)
#     if i>=55:
#         ageGroupTest.append(2)
# deneme_test['Age'] = ageGroupTest


# y=deneme_train["Survived"]
# x=deneme_train.drop(labels=["Survived"],axis=1,inplace=False) #x -> feature(özellik)
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=15)




# print(train_DF.corr()["Survived"].sort_values())
# Sex        -0.536762
# Pclass     -0.356462
# Embarked   -0.181979
# Age        -0.082446
# SibSp      -0.015523
# Parch       0.095265
# Survived    1.000000