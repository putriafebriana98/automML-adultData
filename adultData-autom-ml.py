#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
 

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.dataset import Dataset
from azureml.train.automl import AutoMLConfig
from sklearn.model_selection import train_test_split


# In[2]:


print("This notebook was created using version 1.5.0 of the Azure ML SDK")
print("You are currently using version", azureml.core.VERSION, "of the Azure ML SDK")


# In[3]:


ws = Workspace.from_config()

# Choose a name for the experiment.
experiment_name = 'adultData-automl-regression'

experiment = Experiment(ws, experiment_name)

output = {}
output['Subscription ID'] = ws.name
output['Workspace'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Run History Name'] = experiment_name
pd.set_option('display.max_colwidth', -1)
outputDf = pd.DataFrame(data = output, index = [''])
outputDf.T


# In[4]:


from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your CPU cluster
cpu_cluster_name = "reg-clustera"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                           max_nodes=4)
    compute_target = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)


# In[11]:



df=pd.read_csv('data/adult1000.csv')
#print(df["level"][0]==" Private")
#df=df[df.level.str.strip()=="?"]
df = df[df["level"].str.strip()!="?"]
df = df[df["occupy"].str.strip()!="?"]
df = df[df["country"].str.strip()!="?"]
df=df.drop(['homeNumber', 'homeElectricity','shoesNumber'], axis=1, inplace=False)
df=df.replace({'salary':{' <=50K.':1,' >50K.':3}})
df['level']=df['level'].str.strip()
for column in df.columns:
    df[column]=df[column].astype(str).str.strip()
df['salary']=df['salary'].astype(int)
print(df['salary'].dtypes)
print(df.dtypes)
df.to_parquet('data/adult1000_featurization.parquet')
#print(df[df["level"]=="Private"])


# In[15]:


#data = pd.read_csv("adult1000.csv")
#train_data, test_data = train_test_split(data, test_size=0.2)
#print(os.path.join(os.getcwd(),"data/adult1000_featurization.csv'"))
datastore = ws.get_default_datastore()
datastore.upload(src_dir=os.getcwd())
from azureml.data import DataType
data_types = {
       'salary': DataType.to_long()
   }

dataset = Dataset.Tabular.from_parquet_files(datastore.path('data/adult1000_featurization.parquet'))
train_data, test_data = dataset.random_split(percentage=0.8, seed=223)

label = "salary"


# In[31]:


print(dataset.take(3).to_pandas_dataframe().salary)


# In[7]:


from sklearn.model_selection import train_test_split
trainingSet, testSet = train_test_split(df, test_size=0.2)


# In[16]:


automl_settings = {
    "n_cross_validations": 3,
    "primary_metric": 'r2_score',
    "enable_early_stopping": True, 
    "experiment_timeout_hours": 0.3, #for real scenarios we reccommend a timeout of at least one hour 
    "max_concurrent_iterations": 4,
    "max_cores_per_iteration": -1,
    "verbosity": logging.INFO,
}

automl_config = AutoMLConfig(task = 'regression',
                             compute_target = compute_target,
                             training_data = train_data,
                             label_column_name = label,
                             **automl_settings
                            )


# In[17]:


remote_run = experiment.submit(automl_config, show_output = False)


# In[18]:


remote_run


# In[19]:


from azureml.widgets import RunDetails
RunDetails(remote_run).show()


# In[21]:


remote_run.wait_for_completion()


# In[22]:


best_run,fitted_model=remote_run.get_output()
print(best_run)
print(fitted_model)


# In[23]:


lookup_metric = "root_mean_squared_error"
best_run, fitted_model = remote_run.get_output(metric = lookup_metric)
print(best_run)
print(fitted_model)


# In[24]:


iteration=3
third_run, third_model = remote_run.get_output(iteration = iteration)
print(third_run)
print(third_model)


# In[41]:



import pandas as pd
import numpy as np

df = pd.DataFrame(np.arange(12).reshape(3, 4),
                  columns=['A', 'B', 'C', 'D'])

df.drop([0], axis=0)


# In[42]:


test_data=test_data.to_pandas_dataframe()
y_test=test_data['salary'].fillna(0)
test_data=test_data.drop('salary',1)
test_data=test_data.fillna(0)


# In[43]:


train_data = train_data.to_pandas_dataframe()
y_train = train_data['salary'].fillna(0)
train_data = train_data.drop('salary', 1)
train_data = train_data.fillna(0)


# In[44]:


y_pred_train = fitted_model.predict(train_data)
y_residual_train = y_train - y_pred_train

y_pred_test = fitted_model.predict(test_data)
y_residual_test = y_test - y_pred_test


# In[52]:


print(y_pred_train)


# In[45]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import mean_squared_error, r2_score

# Set up a multi-plot chart.
f, (a0, a1) = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[1, 1], 'wspace':0, 'hspace': 0})
f.suptitle('Regression Residual Values', fontsize = 18)
f.set_figheight(6)
f.set_figwidth(16)

# Plot residual values of training set.
a0.axis([0, 360, -100, 100])
a0.plot(y_residual_train, 'bo', alpha = 0.5)
a0.plot([-10,360],[0,0], 'r-', lw = 3)
a0.text(16,170,'RMSE = {0:.2f}'.format(np.sqrt(mean_squared_error(y_train, y_pred_train))), fontsize = 12)
a0.text(16,140,'R2 score = {0:.2f}'.format(r2_score(y_train, y_pred_train)),fontsize = 12)
a0.set_xlabel('Training samples', fontsize = 12)
a0.set_ylabel('Residual Values', fontsize = 12)

# Plot residual values of test set.
a1.axis([0, 90, -100, 100])
a1.plot(y_residual_test, 'bo', alpha = 0.5)
a1.plot([-10,360],[0,0], 'r-', lw = 3)
a1.text(5,170,'RMSE = {0:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_pred_test))), fontsize = 12)
a1.text(5,140,'R2 score = {0:.2f}'.format(r2_score(y_test, y_pred_test)),fontsize = 12)
a1.set_xlabel('Test samples', fontsize = 12)
a1.set_yticklabels([])

plt.show()


# In[46]:


get_ipython().run_line_magic('matplotlib', 'inline')
test_pred = plt.scatter(y_test, y_pred_test, color='')
test_test = plt.scatter(y_test, y_test, color='g')
plt.legend((test_pred, test_test), ('prediction', 'truth'), loc='upper left', fontsize=8)
plt.show()


# In[ ]:




