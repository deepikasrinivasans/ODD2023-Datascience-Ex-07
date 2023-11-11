# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

## Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

## ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file

## CODE
#### DEVELOPED BY: Deepika S
#### Register no:212222230028

#### Importing library
```
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
```
#### Data loading
```
data = pd.read_csv('/content/titanic_dataset.csv')
data
data.tail()
data.isnull().sum()
data.describe()
```
#### Now, we are checking start with a pairplot, and check for missing values
```
sns.heatmap(data.isnull(),cbar=False)
```
#### Data Cleaning and Data Drop Process
```
data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
data['Age'] = data['Age'].fillna(data['Age'].dropna().median())
```
#### Change to categoric column to numeric
```
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1
```
#### Instead of nan values
```
data['Embarked']=data['Embarked'].fillna('S')
```
#### Change to categoric column to numeric
```
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2
```
#### Drop unnecessary columns
```
drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)
data.head(11)
```
##### Heatmap for train dataset
```
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
```
#### Now, data is clean and read to a analyze
```
sns.heatmap(data.isnull(),cbar=False)
```
#### How many people survived or not... %60 percent died %40 percent survived
```
fig = plt.figure(figsize=(18,6))
data.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()
```
#### Age with survived
```
plt.scatter(data.Survived, data.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()
```
#### Count the pessenger class
```
fig = plt.figure(figsize=(18,6))
data.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = data.drop("Survived",axis=1)
y = data["Survived"]

mdlsel = SelectKBest(chi2, k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
```
#### Split the data into training and test sets
```
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
```
#### Create a Random Forest classifier
```
my_forest = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5, criterion='entropy')
```
#### Fit the model to the training data
```
my_forest.fit(X_train, y_train)
```
#### Make predictions on the test data
```
target_predict = my_forest.predict(X_test)
```
#### Evaluate the model's performance
```
accuracy = accuracy_score(y_test, target_predict)
mse = mean_squared_error(y_test, target_predict)
r2 = r2_score(y_test, target_predict)

print("Random forest accuracy: ", accuracy)
print("Mean Squared Error (MSE): ", mse)
print("R-squared (R2) Score: ", r2)
```
# OUPUT

#### Initial data
![ds1](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex-07/assets/119393935/0c7ea6a1-3af6-4da3-a768-45d8bf28b5b5)
#### Null values
![ds2](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex-07/assets/119393935/ba019ad1-e3e9-4860-9abf-4b339cdf8c0e)
#### Describing the data
![ds3](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex-07/assets/119393935/c7d497e8-6bb5-484f-bfb7-79839fd2160c)
#### Missing values
![ds4](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex-07/assets/119393935/3e17a08d-37e3-4a83-b50e-75deee9921ff)
#### Data after cleaning
![ds5](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex-07/assets/119393935/64aaedae-1f96-4494-a37f-611c7105e046)
#### Data on Heatmap
![ds6](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex-07/assets/119393935/37b4d771-c859-4301-90ac-8e5d395dd80b)
#### Report of(people survied & died)
![ds7](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex-07/assets/119393935/d6057075-5667-4489-acd0-a69ab957f0b6)
#### Cleaned null values
![ds8](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex-07/assets/119393935/20ced343-c22b-42fd-b1e3-3d3452a6eb2c)
#### Report of survied people's age
![ds9](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex-07/assets/119393935/a534944f-a851-49d0-9257-96de303b7ad0)
#### Report of pessengers
![ds10](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex-07/assets/119393935/5b0a25a5-d406-4f59-8db8-fa8174f104ae)
#### Report
![ds11](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex-07/assets/119393935/38654355-ac3f-472a-85d2-8930ffc4a220)

## RESULT:
Thus, Sucessfully performed the various feature selection techniques on a given dataset.
