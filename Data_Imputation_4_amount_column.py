#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Contexto: Revisión de una imputación de datos con el uso de un data set

#importacion de la libreria de pandas para analisis y manipulacion de datos estructurados

#importacion de la libreria de numpy para trabajar con operaciones numericas

import pandas as pd
import numpy as np

#creacion de un dataframe mediante el uso de un diccionario para asignacion de las columnas : id, area, age, amount


data= pd.DataFrame({'id': [1,2,3,4,5,6,7,8,9],
                    'area': [1,2,3,1,2,3,1,2,3],
                    'age': [32,30,40.32,23,27,44,67,23,32],
                    'amount':[102,121,343,np.nan,121,np.nan,155,149,221]
                   })


# In[16]:


# Exploracion de dataframe

data


# In[17]:


#La columna amount es la unica que contiene valores perdidos.


# In[18]:


#Comenzamos la limpieza de datos con la eliminacion de las filas que tienen datos faltantes (indices 3 y 5 )
data1=data.dropna()


# In[19]:


#Exploramos nuevo data set
data1


# In[20]:


#Se confirma la eliminacion de filas con valores faltantes


# In[21]:


#Se genera un nuevo objeto que contenga el promedio de la columna amount (con el nuevo conjunto de datos, el cual no contiene valores perdidos)
mean=data1['amount'].mean()
#visualizar el valor de la media
mean


# In[22]:


# Validamos la media
data1.describe()


# In[23]:


#Se reemplazan los datos faltantes (en la columna amount del data set original) por la media.
data['amount']= data['amount'].fillna(mean)


# In[25]:


#Visualizamos y validamos que los datos perdidos en la columna amount se encuentren reemplazados por la media calculada

data['amount']


# In[26]:


#Finalmente exploramos todo el dataset
data


# In[ ]:





# In[ ]:





# In[ ]:




