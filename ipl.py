#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')


# In[3]:


match_df=pd.read_csv("data/matches.csv")
match_df


# In[4]:


match_df.sample(10)


# In[5]:


match_df.head(10)


# In[6]:


dlvr_df=pd.read_csv("data/deliveries.csv")
dlvr_df.head()


# In[7]:


match_df.info()


# In[8]:


# data analysis


# In[9]:


match_df['winner'].value_counts()


# In[10]:


match_df['team1'].value_counts()


# In[11]:


# check miss value


# In[12]:


match_df[match_df['winner'].isnull() == True]


# In[13]:


# replace


# In[14]:


match_df['winner'].fillna('Draw',inplace=True)


# In[15]:


match_df.info()


# In[16]:


team_encodings={
    'Mumbai Indians': 1,
    'Kolkata Knight Riders': 2,
    'Royal Challengers Bangalore': 3,
    'Deccan Chargers':4,
    'Chennai Super Kings':5,
    'Rajasthan Royals':6,
    'Delhi Daredevils':7,
    'Delhi Capitals':7,
    'Gujarat Lions':8,
    'Kings XI Punjab':9,
    'Sunrisers Hyderabad':10,
    'Rising Pune Supergiants':11, 
    'Rising Pune Supergiant':11,
    'Kochi Tuskers Kerala':12,
    'Pune Warriors':13,
    'Draw':14
}

team_encode_dict ={
    'team1': team_encodings,
    'team2': team_encodings,
    'toss_winner': team_encodings,
    'winner': team_encodings
}
match_df.replace(team_encode_dict,inplace= True)
match_df.head(10)


# In[17]:


##exploring missing value in city


# In[18]:


match_df['city'].value_counts()


# In[19]:


match_df[match_df['city'].isnull()==True ]


# In[20]:


match_df['city'].fillna('Dubai',inplace=True)


# In[21]:


match_df.head()


# In[22]:


match_df.info()


# In[23]:


match_df.describe()


#   Toss match correlation
#   

# In[24]:


toss_wins=match_df['toss_winner'].value_counts(sort=True)
match_wins= match_df['winner'].value_counts(sort=True)

for idx,val in match_wins.iteritems():
    print(f"{list(team_encode_dict['winner'].keys())[idx-1]}-> {toss_wins[idx]}")
    


# In[25]:


match_df['winner'].hist(bins=50)


# In[26]:


fig= plt.figure(figsize=(8,4))
ax1=fig.add_subplot(121)
ax1.set_xlabel('Team')
ax1.set_ylabel('count of toss wins')
ax1.set_title("toss winners")
toss_wins.plot(kind='bar')

ax2=fig.add_subplot(122)
match_wins.plot(kind='bar')

ax2.set_xlabel('Team')
ax2.set_ylabel('count of match wins')
ax2.set_title("match winners")


# In[27]:


match_df.isnull().sum()


# ##drop redundant column

# In[28]:


match_df=match_df[['team1','team2','city','toss_decision','toss_winner','venue','winner']]
match_df


# In[29]:


from sklearn.preprocessing import LabelEncoder

ftr_list=['city','toss_decision','venue']
encoder=LabelEncoder()
for ftr in ftr_list:
    match_df[ftr]=encoder.fit_transform(match_df[ftr])
    print(encoder.classes_)
    
match_df


# ##machine learning

# In[30]:


from sklearn.model_selection import train_test_split
train_df,test_df=train_test_split(match_df, test_size=0.2,random_state=5)
print(train_df.shape)
print(test_df.shape)


# In[31]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def print_model_scores(model,data,predictor,target):
    model.fit(data[predictor],data[target])
    predictions= model.predict(data[predictor])
    accuracy=accuracy_score(predictions,data[target])
    print('Accuracy  %s ' % ' {0:.2}'.format(accuracy))
    scores = cross_val_score(model,data[predictor],data[target],scoring="neg_mean_squared_error",cv=5  )
    print('cross-valid scores:{}'.format(np.sqrt(-scores)))
    print(f'avg rsme: {np.sqrt(-scores).mean()}')
    


# In[32]:


target_var=['winner']
predictor_var=['team1','team2','venue','toss_winner','city','toss_decision']
model=LogisticRegression()
print_model_scores(model,train_df,predictor_var,target_var)


# In[33]:


##random forest
model=RandomForestClassifier(n_estimators=100)
print_model_scores(model,train_df,predictor_var,target_var)
##better accuracy


# In[34]:


team1='Mumbai Indians'
team2='Sunrisers Hyderabad'
toss_winner='Sunrisers Hyderabad'
inp=[team_encode_dict['team1'][team1],team_encode_dict['team2'][team2],'14',team_encode_dict['toss_winner'][toss_winner],'2','1']
inp=np.array(inp).reshape((1,-1))##make 2d array
print(inp)
output=model.predict(inp)
print(output)
print(f"winner:{list(team_encodings.keys())[list(team_encode_dict['team1'].values()).index(output)]}")
      


# In[35]:


##importantance of variables
##depends on what model is used

pd.Series(index=predictor_var,data=model.feature_importances_)


# adding variables

# In[ ]:





# In[ ]:




