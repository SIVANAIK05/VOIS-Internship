 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Create Dataframe for the CSV data


df = pd.read_csv("Attrition_Analytics.csv")
df.head()


# ### Details of the Dataset


print("The Shape of the dataset:",df.shape)


print("No.of Rows in the dataset:",df.shape[0])


print("No.of Columns in the dataset:",df.shape[1])


# ### Statistics of the Dataset


df.describe()


# ### Null Report of the Dataset


print("Nulls present in the dataset column wise")
df.isnull().sum()


print("Total No.of Nulls present in the dataset:",df.isnull().sum().sum())


# ### Display all the Rows having atleast one Null


df[df['YearsWithCurrManager'].isnull()]


# ### Imputing all the nulls with Average value of YearsWithCurrManager Column


df['YearsWithCurrManager'].mean()


df['YearsWithCurrManager'].fillna(round(df['YearsWithCurrManager'].mean()),inplace=True)


# ### Rows with atleast one null


df[df['YearsWithCurrManager'].isnull()]


# ### Note: All Nulls in YearsWithCurrManager Column have been removed


df.isnull().sum()


# ### All Columns of the Dataset


for col in df.columns:
    print(col)


# ### Column's information


df.info()


# ### Column Datatypes


df.dtypes


# ### No.of Numerical Columns of the Dataset


num_numeric_cols = df.select_dtypes(include=['int64','float64'])
print("No.of Numerical Columns in the Dataset:",len(num_numeric_cols.columns))


# ### No.of Categorical Columns of the Dataset


num_categorical_cols = df.select_dtypes(include=['object'])
print("No.of Categorical Columns in the Dataset:",len(num_categorical_cols.columns))


# ### HIstogram for all numerical features to show their distributions


df.hist(bins=10,layout=(6,5),figsize=(10,10))
plt.tight_layout()
plt.show()


# ### Select Important Columns for the Analysis


df.dtypes


selected_cols = ['EmpID','Age','AgeGroup','Attrition','Department','Gender','EducationField','SalarySlab','YearsAtCompany','MonthlyIncome']


df1 = df[selected_cols]
df1.head()


df1['Attrition Count'] = df1['Attrition'].apply(lambda x: 1 if x=='Yes'   else 0)


df1.head()


df1.shape


# ### Outliers Identification


sns.boxplot(y='Age',data=df1)


# ### Note: No Outliers present in Age Column


sns.boxplot(y='YearsAtCompany',data=df1)


# ### Note: Outliers present in YearsAtCompany Column


sns.boxplot(y='MonthlyIncome',data=df1)


# ### Note: Outliers present in MonthlyIncome Column


# ### Remove Outliers


def remove_outlier(col_name):
    sorted(col_name)
    Q1,Q3 = col_name.quantile([0.25,0.75]) 
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound


# ### Removing Outliers for YearsAtCompany Column


low, high = remove_outlier(df1['YearsAtCompany'])


df1['YearsAtCompany'] = np.where(df1['YearsAtCompany']>high,high,df1['YearsAtCompany'])


df1['YearsAtCompany'] = np.where(df1['YearsAtCompany']<low,low,df1['YearsAtCompany'])


sns.boxplot(y='YearsAtCompany',data=df1)


# ### Removing Outliers for MonthlyIncome Column


low , high = remove_outlier(df1['MonthlyIncome'])


df1['MonthlyIncome'] = np.where(df1['MonthlyIncome']>high,high,df1['MonthlyIncome'])


df1['MonthlyIncome'] = np.where(df1['MonthlyIncome']<low,low,df1['MonthlyIncome'])


sns.boxplot(y='MonthlyIncome',data=df1)


df1.head()


# ### Removing All Duplicates


print("No.of Rows present in df1:",df1.shape[0])


df_cleaned = df1.drop_duplicates()


df_cleaned.head()


print("No.of Rows present in df_cleaned:",df_cleaned.shape[0])


print("No.of Duplicate Rows in the final Dataset:",df1.shape[0]-df_cleaned.shape[0])


# ### No.of Duplicate rows are 10 and removed from the dataset. Out final dataset is df_cleaned


# ### Visualization Starts from here


# ### Attrition by Gender using Bar Plot


sns.barplot(x='Gender',y='Attrition Count',data=df_cleaned,estimator=sum)
plt.title('Attrition by Gender')
plt.show()


# ### Note: Male Employees left the Orgnaization more than Female Employees by more than 70%.


# ### Attrition by Education using Pie Chart


df_cleaned['EducationField'].value_counts()


explode = [0.1,0.1,0.1,0.1,0.1,0.1]
plt.figure(figsize=(20,10))
df_cleaned.groupby(['EducationField']).sum().plot(kind='pie',y='Attrition Count',autopct="1.0%%",explode=explode)
plt.title("Attrition by Education")
plt.show()


# ### Note: More Attrition was from Employees with Life Sciences Education


# ### Attrition by Age Group using Column Chart


sns.barplot(x='Attrition Count',y='AgeGroup',data=df_cleaned,estimator=sum)
plt.title('Attrition by AgeGroup')
plt.show()


# ### Note: Employee of 26-35 Years Age Group were left the Organization More. Employees having age more than 55+ were having least attrition 


# ### Attrition by Salary Slab


sns.barplot(x='SalarySlab',y='Attrition Count',data=df_cleaned,estimator=sum)
plt.title('Attrition by Salary Slab')
plt.show()


# ### Note: Employees having Salary less than 5K were left the Organization. Employees having Salary more than 15K were having least attrition.


# ### Attrition by YearsAtCompany


plt.figure(figsize=(10,40))
df_cleaned.groupby(['YearsAtCompany']).sum().plot(kind='line',y='Attrition Count')
plt.title("Attrition by YearsAtCompany")
plt.show()


# ### Note: Most of the Attrition was from the Employees having less than 2.5 Yrs of Experience. Employees having around 12 Yrs of Experience were having least Attrition.


# ### Attrition by Department using Swarm Plot


sns.barplot(x='Department',y='Attrition Count',data=df_cleaned, estimator=sum)
plt.title('Attrition by Department')


# ### Note: Research and Development department had highest Attrition. Human Resources department had least Attrition.


# ### Attrition count


sns.countplot(x='Attrition Count',data=df_cleaned)
plt.title('Attrition vs Stable Employees')


# ### Note: Around 237 Employees were left the Organization


# ### Attrition Percntage


attrition_count = df_cleaned['Attrition Count'].sum()
print("Total Employees left the Organization:",attrition_count)


total_employees = df_cleaned['EmpID'].count()
print("Total No.of Employees in the Organization:",total_employees)


attrition_percent = (attrition_count/total_employees) * 100
print("Attrition Percentage of the Organization:",attrition_percent)


# ### Attrition Percentage of the Organization is around 16%

