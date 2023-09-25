#!/usr/bin/env python
# coding: utf-8

# # 1.Logistic Regression to Predict the Probability of Default of an Individual

# In order to predict a bank loan default, I chose a borrowing dataset that I sourced from Kaggle. This dataset was based on the loans provided to loan applicants. It has many characteristics and my task is to predict loan defaults based on borrower-level features using a multiple logistic regression model

# Probability of default measures the degree of likelihood that the borrower of a loan or debt (the obligor) will be unable to make the necessary scheduled repayments on the debt, thereby defaulting on the debt. Should the obligor be unable to pay, the debt is in default, and the lenders of the debt have legal avenues to attempt a recovery of the debt, or at least partial repayment of the entire debt. The higher the default probability a lender estimates a borrower to have, the higher the interest rate the lender will charge the borrower as compensation for bearing the higher default risk.
# 
# Objective: Come up with a model that can be deployed to predict approval and non approval for new clients

# # 2. Data Understanding

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[2]:


get_ipython().system('pip install imblearn')


# In[3]:


data = pd.read_excel("/Users/arthurkyazze/Desktop/Docs/Kaggle:Github data sets/credit_risk_dataset.xlsx")


# In[4]:


data = data.dropna()
print(data.shape)
print(list(data.columns))


# In[5]:


data.head()


# In[6]:


data['loan_intent'].unique()


# In[7]:


data['loan_status'].value_counts()


# In[8]:


sns.countplot(x='loan_status', data=data, palette='hls')
plt.show()


# In[9]:


count_no_default = len(data[data['loan_status']==0])
count_default = len(data[data['loan_status']==1])
pct_of_no_default = count_no_default/(count_no_default+count_default)
print("\033[1m percentage of no default is", pct_of_no_default*100)
pct_of_default = count_default/(count_no_default+count_default)
print("\033[1m percentage of default", pct_of_default*100)


# In[10]:


data.groupby('loan_status').mean()


# Observations:
# 
#     The average age of loan applicants who defaulted on their loans is slightly less than that of the loan applicants who didn’t.
#     People with more income are less likely to default than those who make less income 
#     People with bigger loans are more likely to default than people with less loans
#     Percentage of income compared to loan also show that people who people with a less percentage of loan compared to income default less compared to those who don't

# In[11]:


data.groupby('person_home_ownership').mean()


# In[12]:


import seaborn as sns
sns.set(style="white")
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0 , 6.0)
sns.kdeplot( data['loan_amnt'].loc[data['loan_status'] == 0], hue=data['loan_status'], shade=True)
sns.kdeplot( data['loan_amnt'].loc[data['loan_status'] == 1], hue=data['loan_status'], shade=True)


# In[13]:


data['loan_amnt'].loc[data['loan_status'] == 0].describe()


# In[14]:


data['loan_amnt'].loc[data['loan_status'] == 1].describe()


# In[15]:


table=pd.crosstab(data.person_home_ownership,data.loan_status)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of person_home_ownership vs Default')
plt.xlabel('person_home_ownership')
plt.ylabel('Proportion of Applicants')
plt.savefig('person_home_ownership_vs_def_stack')


# Person home ownership does seem to be a strong predictor for the target variable

# In[16]:


import seaborn as sns
sns.set(style="white")
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0 , 6.0)
sns.kdeplot( data['loan_int_rate'].loc[data['loan_status'] == 0], hue=data['loan_status'], shade=True)
sns.kdeplot( data['loan_int_rate'].loc[data['loan_status'] == 1], hue=data['loan_status'], shade=True)


# In[17]:


data['loan_int_rate'].loc[data['loan_status'] == 0].describe()


# In[18]:


data['loan_int_rate'].loc[data['loan_status'] == 1].describe()


# In[19]:


import seaborn as sns
sns.set(style="white")
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0 , 6.0)
sns.kdeplot( data['loan_percent_income'].loc[data['loan_status'] == 0], hue=data['loan_status'], shade=True)
sns.kdeplot( data['loan_percent_income'].loc[data['loan_status'] == 1], hue=data['loan_status'], shade=True)


# In[20]:


data['loan_percent_income'].loc[data['loan_status'] == 0].describe()


# In[21]:


data['loan_percent_income'].loc[data['loan_status'] == 1].describe()


# In[22]:


import seaborn as sns
sns.set(style="white")
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0 , 6.0)
sns.kdeplot( data['person_income'].loc[data['loan_status'] == 0], hue=data['loan_status'], shade=True)
sns.kdeplot( data['person_income'].loc[data['loan_status'] == 1], hue=data['loan_status'], shade=True)


# In[23]:


data['person_income'].loc[data['loan_status'] == 0].describe()


# In[24]:


data['person_income'].loc[data['loan_status'] == 1].describe()


# In[25]:


import seaborn as sns
sns.set(style="white")
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0 , 6.0)
sns.kdeplot( data['person_age'].loc[data['loan_status'] == 0], hue=data['loan_status'], shade=True)
sns.kdeplot( data['person_age'].loc[data['loan_status'] == 1], hue=data['loan_status'], shade=True)


# In[26]:


data['person_age'].loc[data['loan_status'] == 0].describe()


# In[27]:


data['person_age'].loc[data['loan_status'] == 1].describe()


# In[28]:


import seaborn as sns
sns.set(style="white")
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0 , 6.0)
sns.kdeplot( data['cb_person_cred_hist_length'].loc[data['loan_status'] == 0], hue=data['loan_status'], shade=True)
sns.kdeplot( data['cb_person_cred_hist_length'].loc[data['loan_status'] == 1], hue=data['loan_status'], shade=True)


# In[29]:


data['cb_person_cred_hist_length'].loc[data['loan_status'] == 0].describe()


# In[30]:


data['cb_person_cred_hist_length'].loc[data['loan_status'] == 1].describe()


# # 3. Data Preparation

# In[31]:


cat_vars=['person_home_ownership']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars=['person_home_ownership']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]


# In[32]:


# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder
le=LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in data.columns.values:
    # Compare if the dtype is object
    if data[col].dtype=='object':
    # Use LabelEncoder to do the numeric transformation
        data[col]=le.fit_transform(data[col])


# In[33]:


data


# In[34]:


# Create feature and target variable for problem
X_class= data.drop('loan_status', axis=1)
y_class = data['loan_status']


# In[35]:


y_class.value_counts()


# In[36]:


# Dealing with imbalanced data set

from imblearn.over_sampling import SMOTE
smote = SMOTE (sampling_strategy='minority')
X_sm, y_sm = smote. fit_resample(X_class,y_class)

y_sm.value_counts()


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


x_train,x_test,y_train,y_test=train_test_split(X_sm,y_sm,train_size=0.75,random_state=15)


# In[39]:


x_train.shape


# Recursive Feature Elimination
# 
# Recursive Feature Elimination (RFE) is based on the idea to repeatedly construct a model and choose either the best or worst performing feature, setting the feature aside and then repeating the process with the rest of the features. This process is applied until all features in the dataset are exhausted. The goal of RFE is to select features by recursively considering smaller and smaller sets of features.

# In[40]:


from sklearn.feature_selection import RFE


# In[41]:


# Create a logistic regression model
model = LogisticRegression()

# Use RFE to select the top 10 features
rfe = RFE(model, n_features_to_select=8)
rfe.fit(x_train,y_train)

# Print the selected features
print(rfe.support_)


# In[42]:


x_train.columns


# In[43]:


data_X1 = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': rfe.ranking_},)
data_X1.sort_values(by=['Importance'])


# In[44]:


cols=[]
for i in range (0, len(data_X1["Importance"])):
    if data_X1["Importance"][i] == 1:
        cols.append(data_X1["Feature"][i])
print(cols)
print(len(cols))


# The RFE has helped us select the following features: 
#     'ID', 'person_age', 'person_income', 'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate

# In[45]:


X=x_train[cols]
y=y_train


# # 4. Modeling

# In[46]:


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# In[47]:


pvalue = pd.DataFrame(result.pvalues,columns={'p_value'},)
pvalue


# In[48]:


pvs=[]
for i in range (0, len(pvalue["p_value"])):
    if pvalue["p_value"][i] < 0.05:
        pvs.append(pvalue.index[i])

if 'const' in pvs:
    pvs.remove('const')
else:
    pvs
print(pvs)
print(len(pvs))


# In[49]:


X=x_train[pvs]
y=y_train

logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())


# In[50]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# # 5. Evaluation

# In[51]:


from sklearn.metrics import accuracy_score
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(accuracy_score(y_test, y_pred)))


# In[52]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[53]:


print("\033[1m The result is telling us that we have: ",(confusion_matrix[0,0]+confusion_matrix[1,1]),"correct predictions\033[1m")
print("\033[1m The result is telling us that we have: ",(confusion_matrix[0,1]+confusion_matrix[1,0]),"incorrect predictions\033[1m")
print("\033[1m We have a total predictions of: ",(confusion_matrix.sum()))


# In[54]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# Interpretations:
# 
#     The precision of class 1 in the test set, that is the positive predicted value of our model, tells us out of all the “bad” loan applicants which our model has identified how many were actually “bad” loan applicants. So, 77% of the “bad” loan applicants which our model managed to identify were actually “bad” loan applicants.
#     The recall of class 1 in the test set, that is the sensitivity of our model, tells us how many “bad” loan applicants our model has managed to identify out of all the “bad” loan applicants existing in our test set. So, our model managed to identify 77% “bad” loan applicants out of all the “bad” loan applicants existing in the test set.

# In[55]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
sns.set(style="whitegrid", color_codes=True)
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# # 6. Deployment

# Finally, the best way to use the model we have built is to assign a probability to default to each of the loan applicant. In order to obtain the probability of probability to default from our model, we will use the following code:

# In[56]:


data['PD'] = logreg.predict_proba(data[X_train.columns])[:,1]
data[['ID', 'PD']].head(10)


# In[57]:


X_train.columns


# #### So, our Logistic Regression model is a good model for predicting the probability of default. Now how do we predict the probability of default for new loan applicant?Suppose there is a new loan applicant, which has: loan_int_rate:12.02, loan_percent_income:0.42, cb_person_default_on_file:Y and person_home_ownership_MORTGAGE:1 . We can take these new data and use it to predict the probability of default for new loan applicant.

# In[58]:


new_data = np.array([12.02,0.42,1,1]).reshape(1, -1)
new_pred=logreg.predict_proba(new_data)[:,1][0]
print("\033[1m This new loan applicant has a {:.2%}".format(new_pred), "chance of paying off the new loan")

