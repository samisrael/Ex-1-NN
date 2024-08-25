<H3>ENTER YOUR NAME</H3>
<H3>ENTER YOUR REGISTER NO.</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

### Import necessary libraries
```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```
### Read the dataset
```python
df = pd.read_csv("Churn_Modelling.csv")
```
### Finding Missing Values
```python
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values)
```
### Handling Missing Values
```python
df.fillna(df.median(), inplace=True)
```
### Checking for duplicates
```python
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

df.drop_duplicates(inplace=True)
s = StandardScaler()
df1= pd.DataFrame(s.fit_transform(df))
```
### Detecting Outliers
```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
outliers_sum = outliers.sum()

print("Number of outliers per column:")
print(outliers_sum)
df1.head()
```
### Normalizing the dataset
```python
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns

scaler = MinMaxScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
X = df1.iloc[:,:-1].values
X
```
### Spliting the dataset into input and output
```python
X = df.drop('Exited', axis=1)
y = df['Exited']
```
### Splitting the data for Training & Testing
```python
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
```
### Printing the training and testing data
```python
print(f"X_train \n {X_train}")
print(f"Length of X_train - {len(X_train)}")
print(f"X_test \n {X_test}")
print(f"Length of X_test - {len(X_test)}")

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
```
## OUTPUT:
SHOW YOUR OUTPUT HERE


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


