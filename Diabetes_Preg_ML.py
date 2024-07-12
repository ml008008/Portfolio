#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Source:https://www.kaggle.com/datasets/mathchi/diabetes-data-set

#(a) Original owners: National Institute of Diabetes and Digestive and
#Kidney Diseases
#(b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu)
#Research Center, RMI Group Leader
#Applied Physics Laboratory
#The Johns Hopkins University
#Johns Hopkins Road
#Laurel, MD 20707
#(301) 953-6231
#(c) Date received: 9 May 1990

#Context:

#This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases.
#The objective is to predict based on diagnostic measurements whether a patient has diabetes.

#Content

#Several constraints were placed on the selection of these instances from a larger database. In particular, 
#all patients here are females at least 21 years old of Pima Indian heritage.


# In[2]:


#   Pregnancies: Number of times pregnant
#   Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#   BloodPressure: Diastolic blood pressure (mm Hg)
#   SkinThickness: Triceps skin fold thickness (mm)
#   Insulin: 2-Hour serum insulin (mu U/ml)
#   BMI: Body mass index (weight in kg/(height in m)^2)
#   DiabetesPedigreeFunction: Diabetes pedigree function
#   Age: Age (years)
#   Outcome: Class variable (0 or 1) 





# In[3]:


#Paquetes Numericos
import numpy as np

#DataFrame Procesamiento
import pandas as pd

#Graficas
import matplotlib.pyplot as plt


# In[4]:


# Leer y cargar el CSV

data= pd.read_csv("https://raw.githubusercontent.com/ml008008/Repo1/main/diabetes.xls")

data.head()


# In[7]:


data.describe()


# In[ ]:


1. Pregnancies
	○ count: 768 women
	○ mean: Average number of pregnancies is about 3.85
	○ std: Standard deviation is about 3.37
	○ min: Minimum number of pregnancies is 0
	○ 25%: 25% of women have 1 or fewer pregnancies
	○ 50%: Median number of pregnancies is 3
	○ 75%: 75% of women have 6 or fewer pregnancies
	○ max: Maximum number of pregnancies is 17
2. Glucose
	○ count: 768 measurements
	○ mean: Average glucose level is about 120.89
	○ std: Standard deviation is about 31.97
	○ min: Minimum glucose level is 0 (likely missing or erroneous data)
	○ 25%: 25% of glucose levels are 99 or lower
	○ 50%: Median glucose level is 117
	○ 75%: 75% of glucose levels are 140.25 or lower
	○ max: Maximum glucose level is 199
3. BloodPressure
	○ count: 768 measurements
	○ mean: Average diastolic blood pressure is about 69.11 mm Hg
	○ std: Standard deviation is about 19.36
	○ min: Minimum blood pressure is 0 (likely missing or erroneous data)
	○ 25%: 25% of blood pressures are 62 or lower
	○ 50%: Median blood pressure is 72 mm Hg
	○ 75%: 75% of blood pressures are 80 or lower
	○ max: Maximum blood pressure is 122 mm Hg
4. SkinThickness
	○ count: 768 measurements
	○ mean: Average skin thickness is about 20.54 mm
	○ std: Standard deviation is about 15.95
	○ min: Minimum skin thickness is 0 (likely missing or erroneous data)
	○ 25%: 25% of skin thickness measurements are 0 or lower
	○ 50%: Median skin thickness is 23 mm
	○ 75%: 75% of skin thickness measurements are 32 or lower
	○ max: Maximum skin thickness is 99 mm
5. Insulin
	○ count: 768 measurements
	○ mean: Average insulin level is about 79.80 mu U/ml
	○ std: Standard deviation is about 115.24
	○ min: Minimum insulin level is 0 (likely missing or erroneous data)
	○ 25%: 25% of insulin levels are 0 or lower
	○ 50%: Median insulin level is 30.5 mu U/ml
	○ 75%: 75% of insulin levels are 127.25 or lower
	○ max: Maximum insulin level is 846 mu U/ml
6. BMI
	○ count: 768 measurements
	○ mean: Average BMI is about 31.99
	○ std: Standard deviation is about 7.88
	○ min: Minimum BMI is 0 (likely missing or erroneous data)
	○ 25%: 25% of BMIs are 27.3 or lower
	○ 50%: Median BMI is 32
	○ 75%: 75% of BMIs are 36.6 or lower
	○ max: Maximum BMI is 67.1
7. DiabetesPedigreeFunction
	○ count: 768 measurements
	○ mean: Average diabetes pedigree function is about 0.47
	○ std: Standard deviation is about 0.33
	○ min: Minimum value is 0.078
	○ 25%: 25% of values are 0.24375 or lower
	○ 50%: Median value is 0.3725
	○ 75%: 75% of values are 0.62625 or lower
	○ max: Maximum value is 2.42
8. Age
	○ count: 768 measurements
	○ mean: Average age is about 33.24 years
	○ std: Standard deviation is about 11.76
	○ min: Minimum age is 21 years
	○ 25%: 25% of ages are 24 or younger
	○ 50%: Median age is 29 years
	○ 75%: 75% of ages are 41 or younger
	○ max: Maximum age is 81 years
9. Outcome
	○ count: 768 outcomes
	○ mean: Average outcome is about 0.35 (indicating that 35% of the cases are positive for diabetes)
	○ std: Standard deviation is about 0.48
	○ min: Minimum outcome is 0 (no diabetes)
	○ 25%: 25% of the outcomes are 0
	○ 50%: Median outcome is 0 (no diabetes)
	○ 75%: 75% of the outcomes are 1 (indicating that 25% of the cases are positive for diabetes)
	○ max: Maximum outcome is 1 (diabetes)
These statistics give a good summary of the distribution of each feature in your dataset. The presence of minimum values of 0 for glucose, blood pressure, skin thickness, insulin, and BMI may indicate missing or erroneous data that should be addressed in data preprocessing.



# In[10]:


data.dtypes


# In[12]:


data.isna() 


# In[13]:


data.isna().sum() 


# In[22]:


#Ajustar el tama;o de la fuente
plt.rcParams['font.size'] = 15

#Crear una figura y ajusta su tama;o
f=plt.figure(figsize=(8,4))

#Crear un subplot o subtrama - al ser una sola figura es 1,1,1

ax= f.add_subplot(1,1,1)

#Grafica tus datos usando 'hist'. Paasa el objeto 'ax' a Pandas. Agrega un borde negro con un #groso de 2.
data["Outcome"].hist(ax=ax, edgecolor='black', linewidth=2)

#Establece los limites en el eje x
ax.set_xlim([-0.5, 1.5])

#Establece la frecuencia de tick. Tenemos 0 y 1 que coresponden a Si y No respectivamente.
ax.set_xticks([0,1])

#Etiquetar xtick labels.
ax.set_xticklabels(["N","Y"])

#Crea el titulo
ax.set_title("Diabetes Y/N?")


# In[23]:


#Creacion de histograma

#Establece la etiqueta del eje Y
ax.set_ylabel("Count")

#Establece los limites superior/inferior del eje y
ax.set_ylim([0,510])

#Hace que las cosas sean bonitas, no es necesario pero se ajusta al tama;o de la figuta
f.tight_layout()


# In[26]:


f=plt.figure(figsize=(8,4))
ax=f.add_subplot(1,1,1)
data["Age"].hist(ax=ax, edgecolor="black", linewidth=2)
ax.set_title("Age range of patients")
ax.set_ylim([0,510])
ax.set_xlabel("Age")
ax.set_ylabel("Count")
f.tight_layout()


# In[ ]:




