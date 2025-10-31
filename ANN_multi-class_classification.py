#!/usr/bin/env python
# coding: utf-8

# ## __Artificial Neural Network-Based Multiclass Classification for Analyzing Aggregate-Processing Dynamics__

# - multiclass classification to:
# >- __v1 dataset__: __normal cells__, cells with __dirupted aggregate movement and coalescence__ and cells with __partially disrupted aggregate movement and coalescence__
# - classification based on combination of __three parameters__: __timepoint__ (minutes), average __size__ of a single aggregate and average __number__ of aggregates per cell
# - data from __quantitative image analyses__ 
# - data entries from __single cells__
# - __classification in training dataset__ based on __experimental evidence__

# ### __Libraries__

# In[648]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pwlf
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from io import BytesIO


# ### __Database and Cloud connection__

# - training datasets can be loaded either from __Local MySQL database__ using a defined __stored procedure__ ('stored_procedure') or from __BackBlaze B2 cloud storage__
# - __multiple version of training datasets__ available, __defined by 'stored_procedure'__ (for MySQL load) __or 'file_name'__ (for Backblaze B2 load)
# >- __dataset_v1__: defined by file name __'ANN_multiclass_classification_training_dataset_v1'__ or stored procedure __'p_ANN_multi_class_classification_v1'__ based on comparison of __WT, mutants _'cin8', 'tpm1', 'tpm2', 'myo4', 'she3'___ and __Latrunculin A-exposed cells__

# * __MySQL authentication parameters__

# In[653]:


#mysql connection parameters
username= 'root'
password= 'poef.qve5353'
hostname= '127.0.0.1'
port= '3306'


# * __establish mysql connection__

# In[656]:


#mysql server connection
connection_string = f"mysql+pymysql://{username}:{password}@{hostname}:{port}/hc_microscopy_data_v2"
engine = create_engine(connection_string) 


# * __used stored procedure__

# In[659]:


stored_procedure= 'p_ANN_multi_class_classification_v1'


# * __Backblaze B2 authentication parameters__

# In[662]:


#Backblaze B2 authentication
data_bucket_name= 'ANN-training-datasets'
bucket_key_id= '003b5f880f95dd40000000008';
bucket_key= 'K003WdKudSgD37pMoUBipXP6nLgGAP0'
file_name= 'ANN_multiclass_classification_training_dataset_v1.csv'


# * __establish Baskblaze B2 connection__

# In[665]:


#connection and authentication
info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", bucket_key_id, bucket_key)
#get the bucket name
bucket = b2_api.get_bucket_by_name(data_bucket_name)


# ### __Data Load__

# * __function__

# In[668]:


def data_load(source= 'backblaze', starting_timepoint= 0, ending_timepoint= 1000):
    
    if source== 'backblaze':

        #data load
        try:
            in_memory_data = BytesIO() #inmemory storage object
            bucket.download_file_by_name(file_name).save(in_memory_data) #download data from specified bucket-file into 'in_memory_storage'
            in_memory_data.seek(0) #rewind to the beginning
            # Load into pandas df
            data = pd.read_csv(in_memory_data)
            data= data.loc[(data.timepoint_minutes >= starting_timepoint) &
                           (data.timepoint_minutes <= ending_timepoint)]
            print('data loaded')
        except Exception as ex:
            print(f'data NOT loaded: {ex}')
            data=  pd.DataFrame()
            
        return data
        
    
    elif source== 'mysql':
        
        #query to obtain the desired data
        query = f"call {stored_procedure} (%s, %s)"
        #data load
        try:
            print('data loaded')
            data= pd.read_sql(query, engine, params= (starting_timepoint, ending_timepoint,))
        except Exception as ex:
            print(f'data NOT loaded: {ex}')
            data=  pd.DataFrame()
        
        return data
    
    else: 
        raise ValueError(f"Invalid source input: '{source}'. Expected: 'backblaze' for cloud or 'mysql' for relational database.")


# * __data load__
# >- to load from local __MySQL__ pass the argument __'mysql'__
# >- to load from __cloud__ pass the argument __'backblaze'__
# >- define __temporal range of included data__ (sequestration stage (movement and coalescence) of aggregate processing), default is the entire dataset

# In[672]:


dataset= data_load(source= 'mysql', starting_timepoint= 0, ending_timepoint= 100)


# In[680]:


dataset.head()


# ### __EDA__

# * __data types__

# In[590]:


dataset.info(memory_usage= 'deep')


# * __missing values__

# In[593]:


dataset.isna().sum()


# * __outlier removal__

# In[596]:


def outlier_removal_iqr(data, col):
    try:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        iqr_filter = (data[col] >= Q1 - .5 * IQR) & (data[col] <= Q3 + .5 * IQR)
        data= data.loc[iqr_filter]
        print('iqr-based outlier removal successful')
    except Exception as ex:
        data= data
        print(f'iqr-based outlier removal NOT successful: {ex}')
    return data


# In[598]:


def outlier_removal_zscore(data, col, thr=1):
    try:
        mean = data[col].mean()
        std = data[col].std(ddof=1)
        z_scores = (data[col] - mean) / std
        zscore_filter = z_scores.abs() <= thr
        data= data.loc[zscore_filter]
        print('zscore-based outlier removal successful')
    except Exception as ex:
        data= data
        print(f'zscore-based outlier removal NOT successful: {ex}')
    return data


# In[600]:


dataset= dataset.groupby('category').apply(lambda x: outlier_removal_iqr(x, 'number_of_foci')).reset_index(drop= True)
dataset= dataset.groupby('category').apply(lambda x: outlier_removal_zscore(x, 'single_focus_avg_area')).reset_index(drop= True)


# * __check datapoints alignemnt__

# In[603]:


tmpts_check= dataset.groupby(['date_label', 'experimental_well_label']).agg({'timepoint_minutes':['min', 'max', 'nunique']})
# tmpts_check


# * __number of data entries per category and timepoint__

# In[606]:


counts_per_category= dataset.groupby('category')[['fov_cell_id']].count().reset_index()
counts_per_category= counts_per_category.assign(category= counts_per_category.category.apply(lambda x: x.replace('_', '\n')))
# counts_per_category


# In[608]:


counts_per_category_per_tmpt= dataset.groupby(['category', 'timepoint_minutes'])[['fov_cell_id']].count().reset_index()
# counts_per_category_per_tmpt


# In[610]:


fig, ax= plt.subplots(1, 2, figsize= (12.8, 4.8))

ax[0].bar(counts_per_category.category,
          counts_per_category.fov_cell_id,
          color= ['red', 'blue', 'green'],
          alpha= .4,
          edgecolor= 'black',
          linewidth= 2)
ax[0].set_ylim(0, 80000)
ax[0].set_title('data-entry count per class',
                weight= 'bold')
ax[0].set_xlabel('category', weight= 'bold')
ax[0].set_ylabel('count', weight= 'bold')

colors= ['red', 'blue', 'green']
for i, category in enumerate(counts_per_category_per_tmpt.category.unique()):
    data_to_plot= counts_per_category_per_tmpt.loc[counts_per_category_per_tmpt.category== category]
    
    ax[1].plot(data_to_plot.timepoint_minutes,
               data_to_plot.fov_cell_id,
               label= category.replace('_', ' '),
               color= colors[i],
               linewidth= 2)
ax[1].set_title('data-entry count per class and timepoint',
                weight= 'bold')
ax[1].legend(frameon= False)
ax[1].set_ylim(0, 5000)
ax[1].set_xlim(220, 400)
ax[1].set_xlabel('timepoint (min)', weight= 'bold')
ax[1].set_ylabel('count', weight= 'bold');


# * __data distributions__

# In[613]:


fig, ax= plt.subplots(3, 3, figsize= (6.4*3, 4.8*3), sharey= 'col', sharex= 'col')

colors= ['red', 'blue', 'green']
for i, category in enumerate(counts_per_category_per_tmpt.category.unique()):
    data_to_plot= dataset.loc[dataset.category== category]
    
    ax[i][0].hist(data_to_plot.timepoint_minutes,
                  bins= 50,
                  color= colors[i],
                  edgecolor= 'white',
                  linewidth= '0.33')
    ax[i][1].hist(data_to_plot.number_of_foci,
                  bins= 50,
                  color= colors[i],
                  edgecolor= 'white',
                  linewidth= '0.33')
    ax[i][2].hist(data_to_plot.single_focus_avg_area,
                  bins= 50,
                  color= colors[i],
                  edgecolor= 'white',
                  linewidth= '0.33')
    ax[i][0].set_ylabel(category.replace('_', '\n'), weight= 'bold')
    ax[i][0].set_xlabel('timepoint (min)', weight= 'bold')
    ax[i][1].set_xlabel('no. of aggregates per cell', weight= 'bold')
    ax[i][2].set_xlabel('average single-aggregate size', weight= 'bold')


# ### __Data Preparation__

# * __remove unneccessary columns__

# In[406]:


dataset= dataset.loc[:, ['timepoint_minutes', 'number_of_foci', 'single_focus_avg_area', 'category']]


# * __split the data into input (x_data) and output (y_data)__

# In[630]:


x_data= dataset.iloc[:, :-1]
# x_data


# In[632]:


y_data= dataset.iloc[:, -1]
# y_data


# * __one-hot encode output (y_data) classes__

# In[635]:


y_data = pd.get_dummies(y_data)


# In[637]:


y_data= y_data.values
# y_data


# * __train vs. test data split__

# In[415]:


x_data_train, x_data_test, y_data_train, y_data_test= train_test_split(x_data, y_data, test_size= 0.25, random_state= 123)


# * __feature scaling__

# In[417]:


scaler= StandardScaler()


# In[418]:


x_data_train= scaler.fit_transform(x_data_train.values)
x_data_test= scaler.transform(x_data_test.values)


# * __final datasets__

# In[420]:


x_data_train


# In[421]:


y_data_train


# ### __Building the ANN__

# * __inititalize the ANN__

# In[424]:


ann = tf.keras.models.Sequential()


# * __first hidden layer__

# In[426]:


ann.add(tf.keras.layers.Dense(units= 6, activation= 'relu'))


# * __second hidden layer__

# In[428]:


ann.add(tf.keras.layers.Dense(units= 6, activation= 'relu'))


# * __output layer__

# In[430]:


ann.add(tf.keras.layers.Dense(units=3, activation='softmax'))


# ### __Compile and Train the ANN__

# * __compile__

# In[433]:


ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# * __train__

# In[435]:


ann.fit(x_data_train, y_data_train, batch_size = 64, epochs = 100)


# ### __Model Evaluation__

# * __predict the test dataset (x_data_test)__

# In[438]:


y_data_pred= ann.predict(x_data_test)
y_data_pred


# * __align the format between y_data_pred and y_data_test (index of the highest probability= category)__

# In[440]:


y_data_pred = np.argmax(y_data_pred, axis=1)
y_data_pred


# In[441]:


y_data_test = np.argmax(y_data_test, axis=1)
y_data_test


# * __accuracy score__

# In[443]:


acc_score= accuracy_score(y_data_test, y_data_pred)
acc_score


# * __confusion matrix__

# In[445]:


cm= confusion_matrix(y_data_test, y_data_pred)
print(cm)


# * __classification report__

# In[447]:


cr= classification_report(y_data_test, y_data_pred)
print(cr)


# ### __Test Prediction for selected mutant__

# * __data load__

# In[521]:


def data_load(mutant, biological_repeat, starting_timepoint= 240, ending_timepoint= 300):
    
    #query to obtain the desired data
    query = "call p_mutant_data_for_predicition (%s, %s, %s, %s)"
    
    #data load
    try:
        print('data loaded')
        data= pd.read_sql(query, engine, params= (mutant, starting_timepoint, ending_timepoint, biological_repeat,))
    except Exception as ex:
        print(f'data NOT loaded: {ex}')
        data=  pd.DataFrame()
        
    return data


# In[523]:


dataset_for_prediction= data_load('MYO4', 1)


# In[525]:


dataset_for_prediction.head()


# * __data cleaning__

# In[528]:


dataset_for_prediction= dataset_for_prediction.groupby('mutation').apply(lambda x: outlier_removal_iqr(x, 'number_of_foci')).reset_index(drop= True)
dataset_for_prediction= dataset_for_prediction.groupby('mutation').apply(lambda x: outlier_removal_zscore(x, 'single_focus_avg_area')).reset_index(drop= True)


# * __data split to control vs. slected mutant and feature scaling__

# In[531]:


control_data= dataset_for_prediction.loc[dataset_for_prediction.mutation== 'wt control']
mutant_data= dataset_for_prediction.loc[dataset_for_prediction.mutation== 'myo4']


# In[533]:


control_dataset_for_prediction= control_data.loc[:, ['timepoint_minutes', 'number_of_foci', 'single_focus_avg_area']]
mutant_dataset_for_prediction= mutant_data.loc[:, ['timepoint_minutes', 'number_of_foci', 'single_focus_avg_area']]


# In[535]:


control_dataset_for_prediction= scaler.transform(control_dataset_for_prediction.values)
mutant_dataset_for_prediction= scaler.transform(mutant_dataset_for_prediction.values)


# * __predictions__

# >- __control__

# In[539]:


control_predictions= ann.predict(control_dataset_for_prediction)


# In[541]:


np.mean(control_predictions, axis= 0)


# >- __mutant__

# In[544]:


mutant_predictions= ann.predict(mutant_dataset_for_prediction)


# In[546]:


np.mean(mutant_predictions, axis= 0)


# In[ ]:





# In[ ]:




