#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Revisión de una imputación de datos con el uso de un data set

import numpy as np
import pandas as pd

data= pd.DataFrame({ 'id':[1,2,3,4,5,6,7,8,9],
                     'area':[1,2,3,1,2,3,1,2,3],
                     'age':[32,30,np.nan,23,27,44,67,23,np.nan],
                     'amount':[102,121,343,np.nan,121,np.nan,155,149,221] 
                   })


# In[19]:


data


# In[20]:


#  Imputación de valores nulos

data.dropna()


# In[21]:


# Este código reducirá la cantidad de datos de la muestra, dejará solo aquellos que son valores numéricos 
# y eliminará las filas en las que se encuentra algún dato faltante.

# Este tipo de imputación es una de las tantas que se pueden realizar.

# También, podemos tener otras opciones, como reemplazar los valores NaN por el promedio de la variable
# que queremos completar. Esto se realiza con la finalidad de no perder datos que sean parte de nuestro modelo.


# In[22]:


# Reemplazo de los valores NaN

import numpy as np
import pandas as pd

data= pd.DataFrame({'id':[1,2,3,4,5,6,7,8,9],
                    'area':[1,2,3,1,2,3,1,2,3],
                    'age':[32,30,np.nan,23,27,44,67,23,np.nan],
                    'amount':[102,121,343,np.nan,121,np.nan,155,149,221]})


# In[23]:


data


# In[24]:


#Se define un nuevo objeto que solo contenga los valores sin NaN
data1= data.dropna()


# In[25]:


data1


# In[26]:


#se aplica el promedio a los datos de edad del objeto data1

mean=data1['age'].mean()
mean



# In[27]:


data1.describe()


# In[28]:


# Se reemplazan los datos de Edad en el data set original

data['age']= data['age'].fillna(mean)


# In[29]:


# Se visualiza el data set completo
data


# In[30]:


# Posteriormente, podemos realizar el mismo procedimiento para completar los datos de la columna amount 

# PENDIENTE CONTINUAR Y COMPLETAR ESTE ULTIMO PROCEDIMIENTO

