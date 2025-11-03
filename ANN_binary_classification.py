### __Artificial Neural Network-Based Binary Classification for Analyzing Aggregate-Processing Dynamics__

# - binary classification to __normal cells__ and cells with __dirupted aggregate movement and coalescence (v1, v2)__, __dirupted aggregate clearance (v3)__ or __lower aggregate formation (v4)__
# - classification based on combination of __three parameters__: __timepoint__ (minutes), average __size__ of a single aggregate and average __number__ of aggregates per cell
# - data from __quantitative image analyses__ 
# - data entries from __single cells__
# - __classification in training dataset__ based on __experimental evidence__



#### __Libraries__
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
# >- __dataset_v1__: defined by file name __'ANN_binary_classification_training_dataset_v1'__ or stored procedure __'p_ANN_binary_classification_v1'__ based on comparison of __control and Latrunculin A-exposed cells__
# >- __dataset_v2__: defined by file name __'ANN_binary_classification_training_dataset_v2'__ or stored procedure __'p_ANN_binary_classification_v2'__ based on comparison of __WT and _tpm1, tpm2_ and _myo4_ mutants__
# >- __dataset_v3__: defined by file name __'ANN_binary_classification_training_dataset_v3'__ or stored procedure __'p_ANN_binary_classification_v3'__ based on comparison of __WT and _ase1, bim1_ and _num1_ mutants__
# >- __dataset_v4__: defined by file name __'ANN_binary_classification_training_dataset_v4'__ or stored procedure __'p_ANN_binary_classification_v4'__ based on comparison of __control and Cycloheximide-exposed cells__

# * __MySQL authentication parameters__
# >- __specify (if applicable)__
#mysql connection parameters
username= ''
password= ''
hostname= ''
port= ''

# * __establish mysql connection__
#mysql server connection
connection_string = f"mysql+pymysql://{username}:{password}@{hostname}:{port}/hc_microscopy_data_v2"
engine = create_engine(connection_string) 

# * __used stored procedure__
stored_procedure= 'p_ANN_binary_classification_v1'
# stored_procedure= 'p_ANN_binary_classification_v2'
# stored_procedure= 'p_ANN_binary_classification_v3'
# stored_procedure= 'p_ANN_binary_classification_v4'

# * __Backblaze B2 authentication parameters__
#Backblaze B2 authentication
data_bucket_name= 'ANN-training-datasets'
bucket_key_id= '003b5f880f95dd40000000008';
bucket_key= 'K003WdKudSgD37pMoUBipXP6nLgGAP0'
file_name= 'ANN_binary_classification_training_dataset_v1.csv'
# file_name= 'ANN_binary_classification_training_dataset_v2.csv'
# file_name= 'ANN_binary_classification_training_dataset_v3.csv'
# file_name= 'ANN_binary_classification_training_dataset_v4.csv'

# * __establish Baskblaze B2 connection__
#connection and authentication
info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", bucket_key_id, bucket_key)
#get the bucket name
bucket = b2_api.get_bucket_by_name(data_bucket_name)



# ### __Data Load__
# * __function__
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
dataset= data_load(source= 'mysql', starting_timepoint= 200, ending_timepoint= 240)
dataset.head()



# ### __EDA__
# * __data types__
dataset.info(memory_usage= 'deep')

# * __missing values__
dataset.isna().sum()
# >- __fill in the NaNs__ for foci __area__ with __0__, only for __v4 data__
dataset= dataset.assign(single_focus_avg_area= dataset.single_focus_avg_area.fillna(0))

# * __outlier removal__
# >- __DO NOT__ apply on the __v4 data__ (aggregate formation analyses)
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

dataset= dataset.groupby('category').apply(lambda x: outlier_removal_iqr(x, 'number_of_foci')).reset_index(drop= True)
dataset= dataset.groupby('category').apply(lambda x: outlier_removal_zscore(x, 'single_focus_avg_area')).reset_index(drop= True)

# * __check datapoints alignemnt__
tmpts_check= dataset.groupby(['date_label', 'experimental_well_label']).agg({'timepoint_minutes':['min', 'max', 'nunique']})
# tmpts_check

# * __number of data entries per category and timepoint__
counts_per_category= dataset.groupby('category')[['fov_cell_id']].count().reset_index()
counts_per_category= counts_per_category.assign(category= counts_per_category.category.apply(lambda x: x.replace('_', '\n')))
# counts_per_category

counts_per_category_per_tmpt= dataset.groupby(['category', 'timepoint_minutes'])[['fov_cell_id']].count().reset_index()
# counts_per_category_per_tmpt

fig, ax= plt.subplots(1, 2, figsize= (12.8, 4.8))
ax[0].bar(counts_per_category.category,
          counts_per_category.fov_cell_id,
          color= ['red', 'blue', 'green'],
          alpha= .4,
          edgecolor= 'black',
          linewidth= 2)
# ax[0].set_ylim(0, 50000)
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
# ax[1].set_ylim(0, 5000)
# ax[1].set_xlim(100, 260)
ax[1].set_xlabel('timepoint (min)', weight= 'bold')
ax[1].set_ylabel('count', weight= 'bold');

# * __data distributions__
fig, ax= plt.subplots(2, 3, figsize= (6.4*3, 4.8*2), sharey= 'col', sharex= 'col')
colors= ['red', 'blue', 'green']
for i, category in enumerate(counts_per_category_per_tmpt.category.unique()):
    data_to_plot= dataset.loc[dataset.category== category]
    
    ax[i][0].hist(data_to_plot.timepoint_minutes,
                  bins= 50,
                  color= colors[i],
                  edgecolor= 'white',
                  linewidth= '0.33')
    ax[i][0].set_xlabel('timepoint (min)', weight= 'bold')
    ax[i][1].hist(data_to_plot.number_of_foci,
                  bins= 50,
                  color= colors[i],
                  edgecolor= 'white',
                  linewidth= '0.33')
    ax[i][1].set_xlabel('no. of aggregates per cell', weight= 'bold')
    ax[i][2].hist(data_to_plot.single_focus_avg_area,
                  bins= 50,
                  color= colors[i],
                  edgecolor= 'white',
                  linewidth= '0.33')
    ax[i][2].set_xlabel('average single-aggregate size', weight= 'bold')
    ax[i][0].set_ylabel(category.replace('_', '\n'), weight= 'bold')



#### __Data Preparation__
# * __remove unneccessary columns__
dataset= dataset.loc[:, ['timepoint_minutes', 'number_of_foci', 'single_focus_avg_area', 'category']]

# * __split the data into input (x_data) and output (y_data)__
x_data= dataset.iloc[:, :-1]
x_data.head()

y_data= dataset.iloc[:, -1]
# y_data

# * __label encoded output (y_data) classes__
label_encoder = LabelEncoder()
y_data= label_encoder.fit_transform(y_data)

# * __train vs. test data split__
x_data_train, x_data_test, y_data_train, y_data_test= train_test_split(x_data, y_data, test_size= 0.25, random_state= 123)

# * __feature scaling__
scaler= StandardScaler()
x_data_train= scaler.fit_transform(x_data_train.values)
x_data_test= scaler.transform(x_data_test.values)

# * __final datasets__
x_data_train
y_data_train



#### __Building the ANN__
# * __inititalize the ANN__
ann = tf.keras.models.Sequential()

# * __first hidden layer__
ann.add(tf.keras.layers.Dense(units= 6, activation= 'relu'))

# * __second hidden layer__
ann.add(tf.keras.layers.Dense(units= 6, activation= 'relu'))

# * __output layer__
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# ### __Compile and Train the ANN__

# * __compile__
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# * __train__
ann.fit(x_data_train, y_data_train, batch_size = 32, epochs = 100)



#### __Model Evaluation__
# * __predict the test dataset (x_data_test)__
y_data_pred= ann.predict(x_data_test)

# * __align the format between y_data_pred and y_data_test (0/1 based on threshold)__
y_data_pred= y_data_pred > 0.5

# * __accuracy score__
acc_score= accuracy_score(y_data_test, y_data_pred)
acc_score

# * __confusion matrix__
cm= confusion_matrix(y_data_test, y_data_pred)
print(cm)

# * __classification report__
cr= classification_report(y_data_test, y_data_pred)
print(cr)

