# Cardio-vascular-disease-prediction
# Cardiovascular-Disease-Prediction
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

#loading data
df=pd.read_csv("cardio_train.csv",delimiter=";")
print(df)
print(df.head())

#print(data.describe())
#checking duplicate values
dup_values=df.duplicated().sum()
print(dup_values)

#checking null values
null_val=df.isnull()
print(null_val)
null_sum=df.isnull().sum()
print(null_sum)

#remove id
drop=df.drop(['id'],axis=1,inplace=True)
print(drop)
print(df.head())

#to get colum age
#age=df['age']
#print(age)

#convert number of days to age
df['age']=(df['age']//365)
print(df)
df.info()


#heatmap on correlation
plt.figure(figsize=(10, 10))
correlation=df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap='Blues')
plt.show()

#create a dataframe based on age
print(df.groupby('age').mean())

plt.figure(figsize=(10, 6))
#plt.bar(mean_cardio_by_age.index, mean_cardio_by_age.values,color='blue')
df.groupby('age')['cardio'].mean().plot(kind='bar')
plt.xlabel('Age')
plt.ylabel('Average Cardio Score')
plt.title('Average Cardio Score by Age')
plt.show()

#counting cardio and non cardio patients
print(df['cardio'].value_counts())
df['cardio'].value_counts().plot(kind='bar')
plt.show()

#to gender counts
gender_counts = df['gender'].value_counts()
print(gender_counts)

#df['gender'].replace({1:'female',2:'male'},inplace=True)
df.groupby('gender')['cardio'].mean().plot(kind='bar')
plt.xlabel('gender')
plt.ylabel('Average Cardio Score')
plt.title('Average Cardio Score by gender')
plt.show()

#cholestrol value indicators
#df['cholesterol']=df['cholesterol'].replace({1: 'normal', 2: 'above normal', 3: 'well above normal' })

# cholesterol count
#cholesterol_counts2 = df['cholesterol'].value_counts()
#print(cholesterol_counts2)

#cobined cholesterol and cardio
#df['cholesterol'].replace({1:'normal',2:'above normal',3:'well above normal'},inplace=True)
grouped_data = df.groupby(['cholesterol', 'cardio']).size().reset_index(name='count')
sns.catplot(x='cholesterol', y='count', hue='cardio',kind='bar', data=grouped_data)
plt.title('Cholesterol Levels among Groups (With/Without Cardio)')
plt.xlabel('Cholesterol Level')
plt.ylabel('Count')
plt.show()



df.plot(kind='box')
plt.show()



#outlier removing
outliers_columns=["height","ap_hi","ap_lo"]
for column in outliers_columns:
 if df[column].dtype in ["int64"]:
   Q1=df[column].quantile(0.25)
   Q3=df[column].quantile(0.75)
   iqr=Q3-Q1
   lower_bound=Q1-1.5*iqr
   upper_bound=Q3+1.5+iqr
   df=df[(df[column]>=lower_bound) & (df[column]<=upper_bound)]
   print(df)

df.plot(kind='box')
plt.show()

#model comparison
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
print(X)
print(y)

models=[]
models.append(('LR',LogisticRegression()))
models.append(('DT',DecisionTreeClassifier()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('GNB',GaussianNB()))
results=[]
names=[]
scoring='accuracy'
kfold=KFold(n_splits=10)
for name,model in models:
	cv_results=cross_val_score(model,X,y,cv=kfold,scoring=scoring)
	results.append(cv_results)
	names.append(name)
	#print(cv_results)
	print(f"Accuracy of {name} is {cv_results.mean()}")
	
fig=plt.figure()
fig.suptitle("Algorithm Comparison")
ax=fig.add_subplot(111)
plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()

#LDA
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model = LinearDiscriminantAnalysis()
model.fit(X_test, y_test)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
print(np.mean(scores))  

new_entry=[58,2,165,76,120,75,3,2,1,1,0]
p=model.predict([new_entry])
print(p)
