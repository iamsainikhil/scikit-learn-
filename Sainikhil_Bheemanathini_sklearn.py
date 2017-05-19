
# coding: utf-8

# In[1]:

##All packages are called in this cell.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[2]:

vpn_df = pd.read_csv('app_usage.csv') #converted the csv file contents into a panda dataframe
vpn_df


# In[3]:

print(vpn_df.shape)
print(vpn_df.columns)
print(vpn_df.head(5))


# In[61]:

X = vpn_df.drop(['RemoteAccess'], axis=1)


# In[58]:

y = vpn_df[['RemoteAccess']]


# In[6]:

ax = sns.distplot(y,bins=10,kde=False) #Used Seaborn to draw histogram plot. Since, I like it.
ax.set(xlabel="VPN Access",ylabel="Numbers")


# In[7]:

corrmat = vpn_df.corr() #corrmat is used to find correlation between features.
f, ax =plt.subplots(figsize=(8,6))
sns.heatmap(corrmat,vmax=.8,square=True,annot=True)
f.tight_layout()


# In[8]:

from sklearn import linear_model


# In[9]:

#The function takes X, y and retrun the trained model and R squared
def train_model(X,y):
    model = linear_model.LinearRegression()
    model.fit(X, y)
    R_2 = model.score(X,y)
    return model, R_2

#function to calculate Adjusted R_square
# n is the number of sample, p is the number features
def cal_adjusted_R(R_2, p, n):
    R_adjusted = R_2-(1-R_2)*(p/(n-p-1))
    return R_adjusted


# In[10]:

R_2_array = np.array([]) #to store the returned R squared values

for col_name in vpn_df.columns:
    if col_name == 'RemoteAccess': #to avoid target column from dataframe
        continue
    else: 
        X_feature = vpn_df[[col_name]] #to extract feature column from dataframe
        
        target = vpn_df[['RemoteAccess']] #y is still the last column
        
        model,R_2 = train_model(X_feature,target) #calling train_model
        
        print(col_name,R_2)
        
        R_2_array = np.append(R_2_array,R_2)      
        
sorted_R_2_index = np.argsort(R_2_array)[::-1] #descending order of sorted R_2_index
    
print("The order of index numbers are : \t", sorted_R_2_index)


# In[11]:

#gradually build up our model and add R squared and adjusted R to the output

for i in range(len(sorted_R_2_index)):
    
    #the selected_features should be the top i most associated features
    selected_features = []
    
    #take the top 1 to ith features as X
    for j in range(i+1):
        
            #append a new column based on the sorted R value
            #take your time to digist this line
            selected_features.append(vpn_df.columns[sorted_R_2_index[j]])
            
    #to verify we got the right features
    print(selected_features)
    
    # X
    X_feature = vpn_df[selected_features]
    
    # y
    target = vpn_df[['RemoteAccess']]
    
    # train the model
    model, R_2 = train_model(X_feature, target)
    
    #to calculate adjusted R
    R_adjusted = cal_adjusted_R(R_2, i+1, vpn_df.shape[0])
    
    #print the output
    print("R2:", R_2, "Ajusted R2:", R_adjusted, )


# In[12]:

#let's build the model with all the features

y = vpn_df['RemoteAccess']
X = vpn_df.drop('RemoteAccess', 1)

from sklearn import linear_model

#create a linear regression model from linear_model package 
model=linear_model.LinearRegression()

#Train the model with our data (X, y)
model.fit(X,y)

#Display the parameters
print('Intercept: \n', model.intercept_)
print('Coefficients: \n', model.coef_)

#use R squared to see how much variation is explained by the trained model
print('R_squared: \n', model.score(X,y))


# In[40]:

X_modified = vpn_df.drop(['RemoteAccess'],1) #To make a dataframe dropping target column
print(X_modified)


# # Using Lasso

# In[50]:

# 1. after reading the above article, you decide to keep only one feature to represent 
# all the features that have correlation higher than 0.9 to it. 

## modify the following code to remove the features you feel necessary
X_modify = vpn_df.drop(['ITOps','Webmail','CloudDrive','RemoteAccess'], 1)

#'ERP','ITOps','Webmail'

# 2. we use Lasso to further penalize models with more features
from sklearn.linear_model import Lasso

# in Lasso, the score is still R squared 
best_score = 0

# Lasso has a parameter alpha used to adjust the level of penalizing the 
# number of features. A bigger alpha will produce less features. 
# We initiate the best alpha to 0 
best_alpha = 0 

# let's fine tune alpha to find the model we need 
for alpha in np.linspace(1,0.2, 1000):
    
    best_score_list = []
    best_alpha_list = []
    for i in range(len(X_modified.columns)):
    #create a linear regression (Lasso) model from linear_model package 
        X_modify = X_modified.iloc[:,i:i+2] #to include all rows and three columns(features) at a time
        model=Lasso(alpha=alpha,normalize=True, max_iter=1e5)
        model.fit(X_modify,y)
        best_score = model.score(X_modify,y)
        best_alpha = alpha
        best_score_list.append(best_score) #to make a list of scores
        best_alpha_list.append(best_alpha) #to make a list of alpha
        
       
    
best_score = max(best_score_list)    #best_score will be the maximum value in the list of scores
print(best_score_list)
print(best_alpha_list)
print("The best R of my 3-feature model is:\t\t", best_score)
print("The alpha I used in Lasso to find my model is: \t", best_alpha)


# # Using Linear Regression

# In[51]:

X_modify_list = []
best_score_list = []
best_alpha_list = []
for i in range(len(X_modified.columns)):
    model = linear_model.LinearRegression()
    X_modify = X_modified.iloc[:,i:i+3] #to include all rows and three columns(features) at a time
    X_modify_list.append(X_modify.columns)
    model.fit(X_modify,y)
    best_score = model.score(X_modify,y)
    best_score_list.append(best_score) #to make a list of scores


print(X_modify_list)
best_score = max(best_score_list)    #best_score will be the maximum value in the list of scores
print(best_score_list)
print("The best R of my 3-feature model is:\t\t", best_score)


# In[57]:

##### Write your summary here
print("My summary:  I choose Linear regression model to be best compared to lasso \n since I am confused with alpha. \n I included the best_score with both the models.\n I don't know where I got wrong with best_aplha. \n Even just by seeing the heatmap one can say that the features with correlation value of above 0.9 \n with respect to target feature 'RemoteAccess'. \n Moreover, I excluded target feature from the model by using X_modified dataframe which doesn't have \n the target feature.")
print("\n\n\The 3 features in my model are: CRM, CloudDrive, ERP ")


# In[ ]:

#Thus, above cells fulfilled Assignment requirements.

