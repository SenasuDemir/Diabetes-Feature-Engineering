####################################
# DIABETES FEATURE ENGINEERING
####################################

#Problem: You are tasked with developing a machine learning model to predict whether individuals are diabetic based on their features.
# Before building the model, you are expected to perform the necessary data analysis and feature engineering steps.

# The dataset is a part of a large dataset maintained by the National Institute of Diabetes and Digestive and Kidney Diseases in the U.S.
# The data was collected for diabetes research on Pima Indian women aged 21 and older, residing in Phoenix, Arizona, the 5th largest city in the state.
# It consists of 768 observations and 8 numerical independent variables.
# The target variable is labeled as "Outcome," where 1 indicates a positive diabetes test result, and 0 indicates a negative result.

# Pregnancies: Number of pregnancies
# Glucose: Glucose level
# BloodPressure: Blood pressure (Diastolic - lower blood pressure)
# SkinThickness: Skin thickness
# Insulin: Insulin level
# BMI: Body Mass Index
# DiabetesPedigreeFunction: A function that calculates the likelihood of diabetes based on family history
# Outcome: Whether the individual is diabetic. (1) for diabetic, (0) for non-diabetic.



# Required Library and Functions
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler,RobustScaler
import warnings
warnings.simplefilter(action='ignore')

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.float_format",lambda x:"%.3f" % x)
pd.set_option("display.width",500)

df=pd.read_csv('datasets/diabetes.csv')


###########################################
# General Picture
###########################################

def check_df(dataframe,head=5):
    print('########################### Shape ###########################')
    print(dataframe.shape)
    print('########################### Types ###########################')
    print(dataframe.dtypes)
    print('########################### Head ###########################')
    print(dataframe.head(head))
    print('########################### Tail ###########################')
    print(dataframe.tail(head))
    print('########################### NA ###########################')
    print(dataframe.isnull().sum())
    print('########################### Quantiles ###########################')
    print(dataframe.quantile([0,0.05,0.50,0.95,0.99,1]).T)

check_df(df)

###########################################
# Define Numeric and Categorical Variables
###########################################

def grab_col_names(dataframe, cat_th=10,car_th=20):
    """
    Define the names of categorical, numerical, and categorical but cardinal variables in the dataset.
    Note: Categorical variables also include those that appear numeric but are actually categorical

    Parameters
    ----------
    dataframe: dataframe
                dataframe from which variable names are to be extracted
    cat_th: int, optional
                Threshold value for the number of classes for variables that are numerical but actually categorical
    car_th: int,optional
                Threshold value for the number of classes for variables that are categorical but cardinal

    Returns
    -------
        cat_cols: list
                List of categorical variables
        num_cols: list
                List of numerical variables
        cat_but_car: list
                List of cardinal variables that appear categorical


    Examples
    --------
        import seaborn as sns
        df=sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        cat_cols + num_cols + cat_but_car = Total number of variables
        num_but_cat is within cat_cols'
    """

    #cat_cols, cat_but_car
    cat_cols=[col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique()< cat_th and dataframe[col].dtypes != 'O']
    cat_but_car=[col for col in dataframe.columns if dataframe[col].nunique()>car_th and dataframe[col].dtypes =='O']
    cat_cols=cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #num_cols
    num_cols=[col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_cols=[col for col in num_cols if col not in num_but_cat]

    print(f'Observations: {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'cat_cols {len(cat_cols)}')
    print(f'num_cols {len(num_cols)}')
    print(f'cat_but_car {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols,num_cols,cat_but_car

cat_cols,num_cols,cat_but_car=grab_col_names(df)


###########################################
# Analysis of Categorical Variables
###########################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        '`Ratio':100 * dataframe[col_name].value_counts()/len(dataframe)}))
    print('**********************************************')
    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show()

cat_summary(df,'Outcome')

###########################################
# Analysis of Numerical Variables
###########################################

def num_summary(dataframe,numerical_col,plot=False):
    quantiles=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df,col,plot=True)


###########################################
# Target Analysis of Numerical Variables
###########################################

def target_summary_with_num(dataframe,target,numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:'mean'}),end='\n\n\n')

for col in num_cols:
    target_summary_with_num(df,'Outcome',col)

###########################################
# Correlation
###########################################

df.corr()

# Correlation Matrix
f,ax=plt.subplots(figsize=[18,13])
sns.heatmap(df.corr(),annot=True, fmt='.2f',ax=ax,cmap='magma')
ax.set_title('Correlation Matrix',fontsize=20)
plt.show()


###########################################
# Base Model Setup
###########################################

y=df['Outcome']
X=df.drop('Outcome',axis=1)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.30,random_state=17)

rf_model=RandomForestClassifier(random_state=46).fit(X_train,y_train)
y_pred=rf_model.predict(X_test)

print(f'Accuracy: {round(accuracy_score(y_pred,y_test),2)}')
print(f'Recall: {round(recall_score(y_pred,y_test),3)}')
print(f'Precision: {round(precision_score(y_pred,y_test),2)}')
print(f'F1: {round(f1_score(y_pred,y_test),2)}')
print(f'Auc: {round(roc_auc_score(y_pred,y_test),2)}' )


# Accuracy: 0.77
# Recall: 0.706
# Precision: 0.59
# F1: 0.64
# Auc: 0.75


def plot_importance(model,features,num=len(X),save=False):
    feature_imp=pd.DataFrame({'Value':model.feature_importances_,'Feature':features.columns})
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x='Value',y='Feature',data=feature_imp.sort_values(by='Value',ascending=False)[0:num])

    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model,X)


###############################################################
# Feature Engineering
###############################################################


###########################################
# Missing Value Analysis
###########################################

# It is known that in a person, the variable values other than Pregnancies and Outcome cannot be 0.
# Therefore, an action decision should be made regarding these values. NaN can be assigned to values that are 0.
zero_columns=[col for col in df.columns if (df[col].min()==0 and col not in ['Pregnancies','Outcome'])]

# We went through each variable with 0 in the observation units and replaced the observation values containing 0 with NaN.
for col in zero_columns:
    df[col]=np.where(df[col]==0,np.nan,df[col])

# Missing Value Analysis
df.isnull().sum()


def missing_values_table(dataframe,na_name=False):
    na_columns=[col for col in dataframe.columns if dataframe[col].isnull().sum()>0]
    n_miss=dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio=(dataframe[na_columns].isnull().sum()/dataframe.shape[0]*100).sort_values(ascending=False)
    missing_df=pd.concat([n_miss,np.round(ratio,2)],axis=1,keys=['n_miss','ratio'])
    print(missing_df,end='\n')
    if na_name:
        return na_columns

na_columns=missing_values_table(df,na_name=True)

# Examining the Relationship Between Missing Values and the Dependent Variable

def  missing_vs_target(dataframe,target,na_columns):
    temp_df=dataframe.copy()
    for col in na_columns:
        temp_df[col+'_NA_FLAG']=np.where(temp_df[col].isnull(),1,0)
    na_flags=temp_df.loc[:,temp_df.columns.str.contains('_NA_')].columns
    for col in na_flags:
        print(pd.DataFrame({'TARGET_MEAN':temp_df.groupby(col)[target].mean(),
                            'COUNT':temp_df.groupby(col)[target].count()}),end='\n\n\n')

missing_vs_target(df,'Outcome',na_columns)

# Filling in Missing Values
for col in zero_columns:
    df.loc[df[col].isnull(),col]=df[col].median()

df.isnull().sum()

###########################################
# Outlier Analysis
###########################################

def outlier_threshold(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe,col_name):
    low_limit, up_limit =outlier_threshold(dataframe,col_name)
    if dataframe[(dataframe[col_name]>up_limit) | (dataframe[col_name]<low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_threshold(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_threshold(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

###########################################
# Outlier Analysis and Suppression Process
###########################################

for col in df.columns:
    print(col,check_outlier(df,col))
    if check_outlier(df,col):
        replace_with_threshold(df,col)

for col in df.columns:
    print(col,check_outlier(df,col))

###########################################
# Feature Extraction
###########################################

# Separating the age variable into categories and creating a new age variable
df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

# BMI below 18.5 is underweight, 18.5 to 24.9 is normal, 24.9 to 29.9 is overweight and over 30 is obese
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Converting glucose value to categorical variable
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

# Creating a categorical variable by considering age and body mass index together
df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"

df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"

df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"

df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

# Creating a categorical variable by dropping Age and Glucose values together
df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"

df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"

df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"

df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"

# Deriving a Categorical Variable with Insulin Value
def set_insulin(dataframe,col_name='Insulin'):
    if 16<=dataframe[col_name]>=166:
        return 'normal'
    else:
        return 'abnormal'
df['NEW_INSULIN_SCORE']=df.apply(set_insulin,axis=1)

df['NEW_GLUCOSE*INSULIN']=df['Glucose']*df['Insulin']
df['NEW_GLUCOSE*PREGNANCIES']=df['Glucose']*df['Pregnancies']

# Enlargement of Columns
df.columns=[col.upper() for col in df.columns]

df.head()
df.shape


###########################################
# Encoding
###########################################

# The process of separating variables according to their types
cat_cols,num_cols,cat_but_car=grab_col_names(df)

# Label Encoding
def label_encoder(dataframe,binary_col):
    labelencoder=LabelEncoder()
    dataframe[binary_col]=labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols=[col for col in df.columns if df[col].dtypes == 'O' and df[col].nunique()==2]

for col in binary_cols:
    df=label_encoder(df,col)

df.head()

# One-Hot Encoding Process
# Updating cat_cols list
cat_cols=[col for col in cat_cols if col not in binary_cols and col not in ['OUTCOME']]


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df=one_hot_encoder(df,cat_cols,drop_first=True)

df.shape
df.head()

###########################################
# Standardizing
###########################################

scaler=StandardScaler()
df[num_cols]=scaler.fit_transform(df[num_cols])

df.head()
df.shape

###########################################
# Modelling
###########################################

y=df['OUTCOME']
X=df.drop('OUTCOME',axis=1)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.30,random_state=17)

rf_model=RandomForestClassifier(random_state=46).fit(X_train,y_train)
y_pred=rf_model.predict(X_test)

print(f'Accuracy: {round(accuracy_score(y_pred,y_test),2)}')
print(f'Recall: {round(recall_score(y_pred,y_test),3)}')
print(f'Precision: {round(precision_score(y_pred,y_test),2)}')
print(f'F1: {round(f1_score(y_pred,y_test),2)}')
print(f'Auc: {round(roc_auc_score(y_pred,y_test),2)}' )

# Accuracy: 0.79
# Recall: 0.72
# Precision: 0.67
# F1: 0.69
# Auc: 0.77


# Base Model
# Accuracy: 0.77
# Recall: 0.706
# Precision: 0.59
# F1: 0.64
# Auc: 0.75

###########################################
# Feature Importance
###########################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)
