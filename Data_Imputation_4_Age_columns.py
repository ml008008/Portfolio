#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Contexto: Revisión de una imputación de datos con el uso de un data set

# numpy and pandas are essential libraries for data manipulation and analysis in Python.
# numpy is often used for numerical operations, while pandas is great for working with structured data (like tables).

import numpy as np
import pandas as pd

#Creating a DataFrame

data= pd.DataFrame({ 'id':[1,2,3,4,5,6,7,8,9],
                     'area':[1,2,3,1,2,3,1,2,3],
                     'age':[32,30,np.nan,23,27,44,67,23,np.nan],
                     'amount':[102,121,343,np.nan,121,np.nan,155,149,221] 
                   })

# The dictionary keys ('id', 'area', 'age', 'amount') become the column names of the DataFrame.
# The lists associated with each key become the rows under the corresponding columns.
# Explanation of Each Column

#     'id': A unique identifier for each record.
#         Values: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#     'area': Represents different areas, possibly a categorical feature.
#         Values: [1, 2, 3, 1, 2, 3, 1, 2, 3]

#     'age': Age of individuals. Notice the np.nan values indicating missing data.
#         Values: [32, 30, np.nan, 23, 27, 44, 67, 23, np.nan]

#     'amount': Some amount associated with each record, possibly monetary. This column also has missing data (np.nan).
#         Values: [102, 121, 343, np.nan, 121, np.nan, 155, 149, 221]

# *************************************************

# Summary

# This code snippet sets up a DataFrame named data with four columns: 'id', 'area', 'age', and 'amount'. 
#     It includes both complete and missing values (np.nan), 
#     simulating a real-world dataset where not all information is always available.


# In[3]:


#Visualize the DataFrame contained in the data variable

data


# Breakdown of the DataFrame

#     Rows and Index:
#         The DataFrame has 9 rows, indexed from 0 to 8.
#         Each row represents an individual record.

#     Columns:
#         'id': An integer identifier for each record.
#         'area': Categorical data representing different areas.
#         'age': Age of individuals, with some missing values (NaN).
#         'amount': Some numerical amount, also with some missing values (NaN).

# Explanation of NaN

#     NaN (Not a Number): Used to represent missing or undefined values in the DataFrame. 
#         In this context, it indicates that the 'age' or 'amount' information is not available for certain records.


# In[4]:


#  Imputación de valores nulos

data.dropna()

# The line of code you provided will drop all rows from the DataFrame data that contain any NaN values.
# Explanation

#     data.dropna(): This function removes any rows in the DataFrame that have one or more "missing (NaN) values".


# In[5]:


# El código anterior "reducirá la cantidad de datos de la muestra", dejará solo aquellos que son valores numéricos 
# y "eliminará las filas en las que se encuentra algún dato faltante".

# Este tipo de imputación es una de las tantas que se pueden realizar.

# También, podemos tener otras opciones, como reemplazar los valores NaN por el promedio de la variable
# que queremos completar. Esto se realiza con la finalidad de no perder datos que sean parte de nuestro modelo.


# In[22]:


# Reemplazo de los valores NaN por el promedio de las variables a completar.

import numpy as np
import pandas as pd

#Generacion de Dataframe, mediante diccionario de datos: columnas or keys son id (unique value), area (categoria de clasificacion de area por codigo).
#age(edad), amount (cantidad)

data= pd.DataFrame({'id':[1,2,3,4,5,6,7,8,9],
                    'area':[1,2,3,1,2,3,1,2,3],
                    'age':[32,30,np.nan,23,27,44,67,23,np.nan],
                    'amount':[102,121,343,np.nan,121,np.nan,155,149,221]})


# In[23]:


data
#Exploracion y visualizacion ordenada de DataFrame


# In[ ]:


# Se puede identificar que tenemos 4 features, columnas, valores de entrada con su respectivo indice.
# Las columnas age y amount contienen valores faltantes , NaN.
# Considerando que el objetivo es reemplazar los datos faltantes con el valor promedio de la variable a la que pertenencen
#Se comienza por limpiar el dataframe (eliminando las filas de los valores faltantes para poder calcular la media de los valores completos dentro del conjunto de datos)
#Se calcula el promedio de la variable que tenia datos faltantes para poder reemplazar NAN por la media.


# In[7]:


#Se define un nuevo objeto que solo contenga los valores sin NaN
#Para realizar esto debemos de generar una nueva variable (objeto) que contenga todos los datos del cojunto(dataframe) que no sean NAN
data1= data.dropna()


# In[15]:


data1


# In[9]:


#se aplica el promedio a los datos de edad del objeto data1 (which contains just the informacion from columna without nan)


Se genera un nuevo objeto que contenga el promedio de la columna edad (sin NAN values)
mean=data1['age'].mean()
mean

# Explanation

#     Calculating the Mean of 'age' Column:
        
#         data1['age']: Selects the 'age' column from the DataFrame data1.
# .mean(): Calculates the mean (average) of the values in the 'age' column.
    
# CALCULATION DETAILS

#     The 'age' column values are: [32, 30, np.nan, 23, 27, 44, 67, 23, np.nan].
#     The mean calculation will ignore NaN values.
#     The valid 'age' values are: [32, 30, 23, 27, 44, 67, 23].

# Mean calculation:
# mean=32+30+23+27+44+67+237=2467≈35.14
# mean=732+30+23+27+44+67+23​=7246​≈35.14

# So, the output will be approximately 35.14.
# Usage

# You can use this mean value to fill in missing values (NaN) in the 'age' column or for other statistical analyses.


# In[11]:


data1.describe()

#     This function generates descriptive statistics that summarize the central tendency, 
#     dispersion, and shape of a dataset’s distribution, excluding NaN values.
    
# he describe() function will provide the following summary statistics for each numerical column in the DataFrame data1:

#     count: Number of non-NaN values.
#     mean: Average of the values.
#     std: Standard deviation, a measure of the amount of variation or dispersion.
#     min: Minimum value.
#     25%: 25th percentile (first quartile).
#     50%: 50th percentile (median).
#     75%: 75th percentile (third quartile).
#     max: Maximum value.

# This summary helps understand the distribution and spread of the data in each numerical column of the DataFrame


# In[12]:


# Se reemplazan los datos de Edad en el data set original

data['age']= data['age'].fillna(mean)

# The code you provided fills the missing values (NaN) in the 'age' column with the mean value of the 'age' column. 
# Let's break down this line of code:


# data['age']: Accesses the 'age' column in the DataFrame data.
# .fillna(mean): Replaces all NaN values in the 'age' column with the calculated mean.
# data['age'] = ...: Assigns the modified column back to data['age']

# The NaN values in the 'age' column have been replaced with the mean value of 35.142857.
# This method ensures that the 'age' column no longer has missing values,
# which is often required for further data analysis or modeling.
    
    


# In[14]:


# Se visualiza el data set completo y se confirma que se aplico el promedio a alos datos de edad del objeto "data1", el cual calculo la media sin valores nan

data


# In[30]:


# Posteriormente, podemos realizar el mismo procedimiento para completar los datos de la columna amount 

# PENDIENTE CONTINUAR Y COMPLETAR ESTE ULTIMO PROCEDIMIENTO

