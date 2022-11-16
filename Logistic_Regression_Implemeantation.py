#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,RidgeCV,LassoCV,ElasticNet,ElasticNetCV,LogisticRegression


# In[2]:


from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score


# In[3]:


import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import seaborn as sns


# In[4]:


import pickle


# In[5]:


df=pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")


# In[6]:


df.head(10)


# In[7]:


df.info()


# In[8]:


ProfileReport(df)


#  ### In our dataset many data have 0 which is not to be possible so we have to replace it by mean value 

# In[9]:


df['BMI']=df['BMI'].replace(0,df['BMI'].mean())


# In[10]:


df.columns


# In[11]:


df['Bloodpressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())


# In[12]:


df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())


# In[13]:


df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())


# In[14]:


df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].mean())


# In[15]:


ProfileReport(df)


# ## Removing the outliers

# In[16]:


fig,ax=plt.subplots(figsize=(20,20))
sns.boxplot(data=df,ax=ax);


# In[17]:


q=df['Insulin'].quantile(.95)
df_new=df[df['Insulin']<95]


# In[18]:


df_new


# In[19]:


fig,ax=plt.subplots(figsize=(20,20))
sns.boxplot(data=df_new,ax=ax);


# In[20]:


q=df['Pregnancies'].quantile(.98)
df_new=df[df['Pregnancies']<q]

q=df_new['BMI'].quantile(.99)
df_new=df_new[df_new['BMI']<q]

q=df_new['SkinThickness'].quantile(.99)
df_new=df_new[df_new['SkinThickness']<q]

q=df_new['Insulin'].quantile(.95)
df_new=df_new[df_new['Insulin']<q]

q=df_new['DiabetesPedigreeFunction'].quantile(.99)
df_new=df_new[df_new['DiabetesPedigreeFunction']<q]


q = df_new['Age'].quantile(.99)
df_new = df_new[df_new['Age']< q]


# In[21]:


fig,ax=plt.subplots(figsize=(20,20))
sns.boxplot(data=df_new,ax=ax);


# In[22]:


ProfileReport(df_new)


# In[24]:


df_new


# In[ ]:



""""def outlier_removal(self,data):
        def outlier_limits(col):
            Q3, Q1 = np.nanpercentile(col, [75,25])
            IQR= Q3-Q1
            UL= Q3+1.5*IQR
            LL= Q1-1.5*IQR
            return UL, LL

        for column in data.columns:
            if data[column].dtype != 'int64':
                UL, LL= outlier_limits(data[column])
                data[column]= np.where((data[column] > UL) | (data[column] < LL), np.nan, data[column])

        #return data """"


# In[25]:


y=df_new['Outcome']
y


# In[26]:


X=df_new.drop(columns=['Outcome'])
X


# ### Data set is fluctutaing alot so we use StandardScaler 

# In[27]:


scalar=StandardScaler()
ProfileReport(pd.DataFrame(scalar.fit_transform(df_new)))


# In[28]:


df_new_scalar=pd.DataFrame(scalar.fit_transform(df_new))
fig,ax=plt.subplots(figsize=(20,20))
sns.boxplot(data=df_new_scalar,ax=ax);


# In[29]:


scalar = StandardScaler()
ProfileReport(pd.DataFrame(scalar.fit_transform(X)))
X_scaled = scalar.fit_transform(X)


# In[30]:


df_new_scalar=pd.DataFrame(scalar.fit_transform(df_new))
fig,ax=plt.subplots(figsize=(20,20))
sns.boxplot(data=df_new_scalar,ax=ax);


# In[31]:


X_scaled


# In[32]:


y


# In[33]:


def vif_score(x):
    scaler = StandardScaler()
    arr = scaler.fit_transform(x)
    return pd.DataFrame([[x.columns[i], variance_inflation_factor(arr,i)] for i in range(arr.shape[1])], columns=["FEATURE", "VIF_SCORE"])


# In[34]:


vif_score(X)


# In[35]:


train_test_split(X_scaled,y,test_size=.20,random_state=144)


# In[36]:


x_train, x_test, y_train, y_test = train_test_split(X_scaled , y , test_size = .20 , random_state = 144)


# In[37]:


x_train


# In[38]:


x_test


# In[39]:


x_test[0]


# In[40]:


logr_liblinear = LogisticRegression(verbose=1,solver='liblinear')


# In[41]:


logr_liblinear.fit(x_train,y_train)


# In[42]:


logr_liblinear.predict_proba([x_test[0]])


# In[43]:


logr_liblinear.predict_proba([x_test[1]])


# In[44]:


logr_liblinear.predict([x_test[1]])


# In[45]:


logr_liblinear.predict_log_proba([x_test[1]])


# In[46]:


type(y_test)


# In[47]:


y_test.iloc[1]


# In[48]:


y_test


# In[49]:


logr = LogisticRegression(verbose=1)


# In[50]:


logr.fit(x_train,y_train)


# In[51]:


logr


# In[52]:


logr_liblinear


# In[53]:


y_pred_liblinear = logr_liblinear.predict(x_test)
y_pred_liblinear


# In[54]:


y_pred_default = logr.predict(x_test)


# In[55]:


y_pred_default


# In[56]:


confusion_matrix(y_test,y_pred_liblinear)


# In[57]:


confusion_matrix(y_test,y_pred_default)


# In[58]:


def model_eval(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    specificity=tn/(fp+tn)
    F1_Score = 2*(recall * precision) / (recall + precision)
    result={"Accuracy":accuracy,"Precision":precision,"Recall":recall,'Specficity':specificity,'F1':F1_Score}
    return result
model_eval(y_test,y_pred_liblinear)


# In[59]:


model_eval(y_test,y_pred_default)


# In[60]:


auc = roc_auc_score(y_test,y_pred_liblinear)


# In[61]:


auc


# In[62]:


roc_auc_score(y_test,y_pred_default)


# In[63]:


fpr, tpr, thresholds  = roc_curve(y_test,y_pred_liblinear)


# In[64]:


fpr


# In[65]:


tpr


# In[66]:


thresholds


# In[67]:


plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




