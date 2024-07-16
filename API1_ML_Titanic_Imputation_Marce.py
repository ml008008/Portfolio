#!/usr/bin/env python
# coding: utf-8

# In[179]:


#API 1 _ MACHINE LEARNING
#AUTHOR: Marce Martinez

#Importacion de libreria Pandas para analisis y manipulacion de data frame.
import pandas as pd

# Lectura del archivo Excel desde la URL y lo carga en un DataFrame llamado data.
url = 'https://github.com/ml008008/Repo1/raw/main/titanic_M1.xlsx'
data = pd.read_excel(url)
data.head()


# In[180]:


#Descripcion de la cantidad de valores faltantes para cada columna:
missing_values = data.isnull().sum()
print("Cantidad de valores faltantes por columna:\n", missing_values)


# In[181]:


#Se completan los valores faltantes en la columna 'Pclass' con 2:
data['pclass'] = data['pclass'].fillna(2)


# In[182]:


#Proceso de verificacion

# a) Verificar la cantidad de valores faltantes en 'Pclass' después de la imputación
missing_pclass_after = data['pclass'].isnull().sum()
print("Cantidad de valores faltantes en 'Pclass' después de la imputación:", missing_pclass_after)


# In[183]:


# b) Verificar que no hay valores faltantes en 'Pclass'
if missing_pclass_after == 0:
    print("Todos los valores faltantes en 'Pclass' han sido reemplazados con 2.")
else:
    print("Aún hay valores faltantes en 'Pclass'.")


# In[184]:


# c) Revisar los valores únicos en la columna 'Pclass'
unique_pclass_values = data['pclass'].unique()
print("Valores únicos en la columna 'Pclass':", unique_pclass_values)


# In[185]:


#Imputacion de valores faltantes en la columna 'sex':

unique_sex_values = data['sex'].unique()
print("Valores únicos en la columna 'sex' antes de la imputación:\n", unique_sex_values)

data['sex'] = data['sex'].fillna('unknown')
data['sex'] = data['sex'].replace('unknown', 'male')
# Muestra los valores únicos en la columna 'Sex' antes de la imputación.
# Rellena los valores faltantes con 'unknown'.
# Reemplaza 'unknown' con 'male'.


# In[186]:


#Se rellenan los valores faltantes en la columna 'Age' con el promedio de edad:
mean_age = data['age'].mean()

print(mean_age)



# In[187]:


#Validacion de calculo de media
data.describe()


# In[188]:


data['age'] = data['age'].fillna(mean_age)
print(data['age'])


# In[189]:


#Validacion de reemplazo de valores faltantes de columna age con media = 30.026275510204076


# c) Verificar la cantidad de valores faltantes en 'Age' después de la imputación
missing_age_after = data['age'].isnull().sum()
print("Cantidad de valores faltantes en 'age' después de la imputación:", missing_age_after)

# d) Asegurarse de que todos los valores faltantes han sido reemplazados
if missing_age_after == 0:
    print("Todos los valores faltantes en 'age' han sido reemplazados con el promedio de edad.")
else:
    print("Aún hay valores faltantes en 'age'.")


# In[190]:


#Exploracion de valores perdidos dentro de embarked andtes de realizar los cambios solicitados
missing_values = data.isnull().sum()
print("Cantidad de valores faltantes por columna antes de la imputación:\n", missing_values)


# In[191]:


#Finalmmente se muestra la asignacion de el valor 's' a los valores faltantes en la columna 'Embarked':
data['embarked'] = data['embarked'].fillna('s')


# In[192]:


#Validacion de cambios aplicados en embarked
# Contar valores faltantes después de la imputación
missing_embarked_after = data['embarked'].isnull().sum()
print("Cantidad de valores faltantes en 'embarked' después de la imputación:", missing_embarked_after)

# Asegurarse de que no hay valores faltantes en 'Embarked'
if missing_embarked_after == 0:
    print("Todos los valores faltantes en 'embarked' han sido reemplazados con 'S'.")
else:
    print("Aún hay valores faltantes en 'embarked'.")

# Revisar los valores únicos en la columna 'Embarked'
unique_embarked_values = data['embarked'].unique()
print("Valores únicos en la columna 'Embarked':", unique_embarked_values)

