#!/usr/bin/env python
# coding: utf-8

# ## __Application of clustering algorithms to set the temporal boundaries of aggregate-processing stages__

# ### __Libraries__

# In[98]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import pwlf
from sqlalchemy import create_engine
from scipy.signal import find_peaks


# ### __Database Authentication__

# * __fill in__ the authentication parameters

# In[7]:


#mysql connection parameters
username= 'root'
password= 'poef.qve5353'
hostname= '127.0.0.1'
port= '3306'

#mysql server connection
connection_string = f"mysql+pymysql://{username}:{password}@{hostname}:{port}/hc_microscopy_data_v2"
engine = create_engine(connection_string) 


# ### __Data Load__

# * data loaded from __'hc_microscopy_data_v2' database__
# * pre-defined __stored procedure__ 'p_TS_screen_WT_sbw_data'
# * complete __control__ (WT Hsp104-GFP) __dataset__
# * __data filtered by__ lower and upper range of __cell counts in the initital timepoint__

# In[10]:


def data_load(min_cell_count, max_cell_count, starting_timepoint):
    
    #query to obtain the desired data
    query = "call p_TS_screen_WT_sbw_data (%s, %s, %s)"
    
    #data load
    try:
        print('data loaded')
        data= pd.read_sql(query, engine, params= (min_cell_count, max_cell_count, starting_timepoint,))
    except Exception as ex:
        print(f'data NOT loaded: {ex}')
        data=  pd.DataFrame()
        
    return data


# * __cell-count range__ for the initital timepoint set to __100-300__ cells
# * above 300 cell counts tend to overgrow in the later timepoints

# In[12]:


data= data_load(100, 300, 4)


# ### __Data Preparation__

# * data __normalized 0-1__
# * __univariete outliers filtered out__ based on __z-score (-3 to 3)__
# * __multivariete outliers filtered out__ using the __DBSCAN clustering__

# * __data parameters__

# In[16]:


data.isna().sum()


# In[17]:


data.info(memory_usage= 'deep')


# In[18]:


data.describe()


# * __data distributions__

# In[20]:


fig, ax= plt.subplots(1, 3, figsize= (19.2, 4.8), sharey= 'all')

ax[0].hist(data.proportion_of_cells_with_agg,
           bins= 50,
           color= 'blue',
           edgecolor= 'white',
           linewidth= '0.33')
ax[0].set_title('Proportion of cells containing aggregates', weight= 'bold')
ax[0].set_ylim(0, 1000)

ax[1].hist(data.avg_number_of_foci_per_cell,
           bins= 50,
           color= 'red',
           edgecolor= 'white',
           linewidth= '0.33')
ax[1].set_title('Average number of aggregates per cell', weight= 'bold')


ax[2].hist(data.avg_size_single_focus,
           bins= 50,
           color= 'green',
           edgecolor= 'white',
           linewidth= '0.33')
ax[2].set_title('Average size of a single aggregate', weight= 'bold');


# * __normalisation and standardisation__

# In[22]:


def _0_1_normalisation(column):
    
    try:
        #0-1 normalisation
        normalised_column= (column-column.min())/(column.max()-column.min())
        return normalised_column
    except Exception as ex:
        print(f'normalization failed: {ex}')
        return column

def standardisation(column):

    try:
        #standardisation
        standardised_column= (column-column.mean())/column.std(ddof= 1)
        return standardised_column
    except Exception as ex:
        print(f'standardization failed: {ex}')
        return column


# In[23]:


data= data.assign(proportion_of_cells_with_agg_norm= _0_1_normalisation(data.proportion_of_cells_with_agg),
                  proportion_of_cells_with_agg_zscore= standardisation(data.proportion_of_cells_with_agg),
                  avg_number_of_foci_per_cell_norm= _0_1_normalisation(data.avg_number_of_foci_per_cell),
                  avg_number_of_foci_per_cell_zscore= standardisation(data.avg_number_of_foci_per_cell),
                  avg_size_single_focus_norm= _0_1_normalisation(data.avg_size_single_focus),
                  avg_size_single_focus_zscore= standardisation(data.avg_size_single_focus))


# * __z-score-based univariete outlier removal (-3 to 3)__

# In[25]:


def outlier_removal(data):
    try:
        data= data.loc[(data.proportion_of_cells_with_agg_zscore >= -3) & (data.proportion_of_cells_with_agg_zscore <= 3) &
                       (data.avg_number_of_foci_per_cell_zscore >= -3) & (data.avg_number_of_foci_per_cell_zscore <= 3) &
                       (data.avg_size_single_focus_zscore >= -3) & (data.avg_size_single_focus_zscore <= 3)].reset_index(drop= True)
        print('data cleaning succesful')
        return data
    except Exception as ex:
        print(f'data cleaning failed: {ex}')
        return data


# In[26]:


data= outlier_removal(data)


# * __DBSCAN-based multivariete outlier removal__

# In[28]:


def dbscan_outlier_detection(data, eps= 0.05, min_samples= 30):
    try:
        dbscan_cleaning= DBSCAN(eps= eps, min_samples= min_samples)
        dbscan_cleaning.fit(data.loc[:, ['proportion_of_cells_with_agg_norm', 'avg_number_of_foci_per_cell_norm', 'avg_size_single_focus_norm']])
        data= data.assign(dbscan_clusters= dbscan_cleaning.labels_)
        data= data.assign(dbscan_outlier= np.where(data.dbscan_clusters==-1, True, False))
        data= data.drop(columns= ['dbscan_clusters'])
        print(f'outlier detection succesful')
        return data
        
    except Exception as ex:
        print(f'outlier detection failed: {ex}')
        data= data.assign(dbscan_outlier= False)
        return data


# In[29]:


data= dbscan_outlier_detection(data)


# In[30]:


fig, ax= plt.subplots(1, 3, figsize= (19.2, 4.8), sharex= 'all', sharey= 'all')

ax[0].scatter(data.avg_number_of_foci_per_cell_norm,
              data.avg_size_single_focus_norm,
              alpha= 0.2,
              edgecolor= 'white',
              linewidth= 0.75,
              color= 'Blue')
ax[0].set_title('full dataset', weight= 'bold')
ax[0].set_xlim(-0.05, 1.05)
ax[0].set_ylim(-0.05, 1.05)

colors= ['Green' if point== False else 'Black' for point in data.dbscan_outlier]
alphas= [0.2 if point== False else 0.5 for point in data.dbscan_outlier]
ax[1].scatter(data.avg_number_of_foci_per_cell_norm,
              data.avg_size_single_focus_norm,
              alpha= alphas,
              edgecolor= 'white',
              linewidth= 0.75,
              color= colors)
ax[1].set_title('multivariete outliers', weight= 'bold')

ax[2].scatter(data.loc[data.dbscan_outlier==False, 'avg_number_of_foci_per_cell_norm'],
              data.loc[data.dbscan_outlier==False, 'avg_size_single_focus_norm'],
              alpha= 0.2,
              edgecolor= 'white',
              linewidth= 0.75,
              color= 'Green')
ax[2].set_title('cleaned dataset', weight= 'bold');


# In[31]:


data= data.loc[data.dbscan_outlier==False]


# * __cleared data distribution__

# In[33]:


fig, ax= plt.subplots(1, 3, figsize= (19.2, 4.8), sharey= 'all')

ax[0].hist(data.proportion_of_cells_with_agg,
           bins= 50,
           color= 'blue',
           edgecolor= 'white',
           linewidth= '0.33')
ax[0].set_title('Proportion of cells containing aggregates', weight= 'bold')
ax[0].set_ylim(0, 1000)

ax[1].hist(data.avg_number_of_foci_per_cell,
           bins= 50,
           color= 'red',
           edgecolor= 'white',
           linewidth= '0.33')
ax[1].set_title('Average number of aggregates per cell', weight= 'bold')


ax[2].hist(data.avg_size_single_focus,
           bins= 50,
           color= 'green',
           edgecolor= 'white',
           linewidth= '0.33')
ax[2].set_title('Average size of a single aggregate', weight= 'bold');


# ### __Clustering__

# * __three__ clustering __algorithms__ applied: __K-Means__, __DBSCAN__ and __Gaussian Mixture Model__
# * for __K-Means__ and __GMM__ the __number of clusters set to 3__
# * __clustering based on three-dimensional data__: percentage of cells with aggregates, aggregate size and aggregate number
# * data and clusters __visualized in two-dimensional space__: aggregate size and aggregate number

# * __data__

# In[37]:


clustering_data= data.loc[:, ['timepoint_minutes', 'proportion_of_cells_with_agg_norm', 'avg_number_of_foci_per_cell_norm', 'avg_size_single_focus_norm']]
clustering_data.head()


# * __k-means__

# In[39]:


def k_means(data, k= 3):
    
    try:
        kmeans_clustering= KMeans(n_clusters= k)
        kmeans_clustering.fit(data.loc[:, ['proportion_of_cells_with_agg_norm', 'avg_number_of_foci_per_cell_norm', 'avg_size_single_focus_norm']])
        clustered_data= data.assign(k_means_clusters= kmeans_clustering.labels_)
        print('clustering succesful')
        return clustered_data
    except Exception as ex:
        print(f'clustering failed: {ex}')
        return data.assign(k_means_clusters= 0)


# In[40]:


kmeans_data= k_means(clustering_data)


# * __dbscan__

# In[42]:


def dbscan(data, epsilon= 0.5, no_of_points= 5):
    
    try:
        dbscan_clustering= DBSCAN(eps= epsilon, min_samples= no_of_points)
        dbscan_clustering.fit(data.loc[:, ['proportion_of_cells_with_agg_norm', 'avg_number_of_foci_per_cell_norm', 'avg_size_single_focus_norm']])
        clustered_data= data.assign(dbscan_clusters= dbscan_clustering.labels_)
        print('clustering succesful')
        return clustered_data
    except Exception as ex:
        print(f'clustering failed: {ex}')
        return data.assign(dbscan_clusters= 0)


# In[43]:


dbscan_data= dbscan(clustering_data, 0.05, 30)


# * __gaussian mixture model__

# In[45]:


def gmm(data, n_clusters= 3):
    
    try:
        gmm_clustering= GaussianMixture(n_components= n_clusters)
        gmm_clustering.fit(data.loc[:, ['proportion_of_cells_with_agg_norm', 'avg_number_of_foci_per_cell_norm', 'avg_size_single_focus_norm']])
        clustered_data= data.assign(gmm_clusters= gmm_clustering.predict(data.loc[:, ['proportion_of_cells_with_agg_norm', 'avg_number_of_foci_per_cell_norm', 'avg_size_single_focus_norm']]))
        print('clustering succesful')
        return clustered_data
    except Exception as ex:
        print(f'clustering failed: {ex}')
        return data.assign(gmm_clusters= 0)


# In[46]:


gmm_data= gmm(clustering_data)


# In[47]:


fig, ax= plt.subplots(1, 3, figsize= (19.2, 4.8), sharey= 'all', sharex= 'all')

ax[0].scatter(kmeans_data.avg_number_of_foci_per_cell_norm,
              kmeans_data.avg_size_single_focus_norm,
              c= kmeans_data.k_means_clusters,
              alpha= 0.15,
              edgecolor= 'white',
              linewidth= 0.75)
ax[0].set_ylim(-0.05, 1.05)
ax[0].set_xlim(-0.05, 1.05)
ax[0].set_title('K-Means clustering', weight= 'bold')

ax[1].scatter(dbscan_data.avg_number_of_foci_per_cell_norm,
              dbscan_data.avg_size_single_focus_norm,
              c= dbscan_data.dbscan_clusters,
              alpha= 0.2,
              edgecolor= 'white',
              linewidth= 0.75)
ax[1].set_title('DBSCAN clustering', weight= 'bold')

ax[2].scatter(gmm_data.avg_number_of_foci_per_cell_norm,
              gmm_data.avg_size_single_focus_norm,
              c= gmm_data.gmm_clusters,
              alpha= 0.15,
              edgecolor= 'white',
              linewidth= 0.75)
ax[2].set_title('GMM clustering', weight= 'bold');


# ### __Pearson's Correlation Coefficient and Regression Line Fitting__

# * __GMM__ selected as __the most suitable__ for temporal boundaries identification
# * __PCC__ calculated and __RL__ fitted for each of the identified stages (clusters)
# * identified clusters __re-clustered by DBSCAN to remove multivariete outliers__ (and limit overlaps between the clusters)

# In[50]:


gmm_data.head()


# * __DBSCAN re-clustering (outlier/cluster-overlap removal)__

# In[52]:


gmm_data= gmm_data.groupby('gmm_clusters').apply(lambda x: dbscan_outlier_detection(x, eps= 0.05, min_samples= 40))


# In[53]:


fig, ax= plt.subplots(1, len(gmm_data.gmm_clusters.unique()), figsize= (6.4*len(gmm_data.gmm_clusters.unique()), 4.8))

cmap = plt.cm.get_cmap('Set1')   
colors = cmap.colors 

for i, cluster in enumerate(gmm_data.gmm_clusters.unique()):
    
    cluster_data= gmm_data.loc[gmm_data.gmm_clusters==cluster]
    alphas= [0.15 if point== False else 0.5 for point in cluster_data.dbscan_outlier]
    outlier_colors= [colors[i] if point== False else 'Black' for point in cluster_data.dbscan_outlier]
    x= cluster_data.avg_number_of_foci_per_cell_norm
    y= cluster_data.avg_size_single_focus_norm
    ax[i].scatter(x,
                  y,
                  alpha= alphas,
                  color= outlier_colors,
                  edgecolor= 'white',
                  linewidth= .75)
    
    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(0, 1)
    ax[i].set_xlabel('average number of aggregates per cell (normalized)', weight= 'bold')
    ax[i].set_ylabel('average size of a single aggregate (normalized)', weight= 'bold')


# In[54]:


gmm_data= gmm_data.loc[gmm_data.dbscan_outlier==False]


# * __linear regression__

# In[56]:


fig, ax= plt.subplots(figsize=(12.8, 9.6))

cmap = plt.cm.get_cmap('Set1')   
colors = cmap.colors 

for i, cluster in enumerate(gmm_data.gmm_clusters.unique()):
    
    color= colors[i]
    cluster_data= gmm_data.loc[gmm_data.gmm_clusters==cluster]
    x= cluster_data.avg_number_of_foci_per_cell_norm
    y= cluster_data.avg_size_single_focus_norm
    cc= round(x.corr(y, method= 'pearson', min_periods= 2), 2)
    
    ax.scatter(x,
               y,
               alpha= 0.25,
               color= colors[i],
               edgecolor= 'white',
               linewidth= 0.75,
               label= f'cc: {cc}')
    
    m, b= np.polyfit(x, y, 1)
    ax.plot(x, m*x+b,
            color= color,
            linewidth= 5)


ax.set_ylim(0, 1)
ax.set_xlim(0, 1)
ax.set_xlabel('average number of aggregates per cell (normalized)', weight= 'bold', fontsize= 12)
ax.set_ylabel('average size of a single aggregate (normalized)', weight= 'bold', fontsize= 12)
ax.set_title('GMM clustering + Linear regression', weight= 'bold')
ax.legend(ncol= 3, frameon= False, loc= 'lower right', prop={'weight': 'bold', 'size': 10});


# * __timepoints overlap__

# In[158]:


gmm_data.head()


# >- __histogram peak for each cluster__

# In[228]:


def peaks(data, min_height= 100):
    try:
        tmpt_counts= data.groupby('timepoint_minutes')[['timepoint_minutes']].value_counts().reset_index()
        peaks, properties = find_peaks(tmpt_counts.iloc[:, 1], height= min_height) 
        peak= tmpt_counts.iloc[peaks].timepoint_minutes.mean()
    except Exception as ex:
        print(f'peak identification FAILED: {ex}')
        peak= 0
    return peak


# >- __visualisation__

# In[236]:


fig, ax= plt.subplots(figsize= (19.2, 4.8))

cmap = plt.cm.get_cmap('Set1')   
colors = cmap.colors 

for i, cluster in enumerate(gmm_data.gmm_clusters.unique()):
    
    cluster_data= gmm_data.loc[gmm_data.gmm_clusters==cluster]
    peak= peaks(cluster_data)

    ax.hist(cluster_data.timepoint_minutes, 
            bins= 50,
            color= colors[i],
            edgecolor= 'white',
            linewidth= '0.33',
            alpha= 0.25)
    ax.axvline(peak, color= colors[i], ls= '--')
    ax.set_title('Timepoint overlaps', weight= 'bold')
    ax.set_ylim(0, 250)
ax.set_xlabel('timepoint (min)', weight= 'bold', fontsize= 12);


# >- __temporal segemntation__
# >>- formation <= Peak 1
# >>- sequestration > Peak 1 and <= Peak 3
# >>- clearance > Peak 3

# In[250]:


gmm_data.groupby('gmm_clusters').apply(lambda x: peaks(x)).sort_values()

