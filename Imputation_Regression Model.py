#!/usr/bin/env python
# coding: utf-8

# In[68]:


# This example demonstrates a straightforward approach to regression imputation.


# In[69]:


import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

# Sample data
data = {
    'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Y': [2, 4, np.nan, 8, 10, 12, np.nan, 16, 18, 20]
}
df = pd.DataFrame(data)

#  This code imports necessary libraries, creates a sample dataset with missing values,
#     and converts it into a pandas DataFrame for further analysis and imputation.


# In[70]:


# Explanation:

#    Import Libraries:
#    import pandas as pd: Imports the pandas library, which is used for data manipulation and analysis.
#    import numpy as np: Imports the numpy library, which is used for numerical operations and handling missing values (np.nan).
#    from sklearn.linear_model import LinearRegression: 
#.   Imports the LinearRegression class from the scikit-learn library, which will be used to create a linear regression model.

#     Sample Data:
#         data = {'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'Y': [2, 4, np.nan, 8, 10, 12, np.nan, 16, 18, 20]}
#             Creates a dictionary with two keys: X and Y.
#             The X key has a list of numbers from 1 to 10.
#             The Y key has a list of numbers with some missing values represented by np.nan.

#     Create DataFrame:
#         df = pd.DataFrame(data)
#             Converts the dictionary data into a pandas DataFrame named df.
#             The resulting DataFrame has two columns: X and Y, with Y containing some missing values (np.nan).


# In[71]:


# Display the original data
print("Original DataFrame:")
print(df)



# In[72]:


# Split the data into training and prediction sets

train_df = df[df['Y'].notna()]
print("This dataframe will be used to train the regression model")
print(train_df)


predict_df = df[df['Y'].isna()]
print("This dataframe will be used for predicting and imputing the missing Y values")
print(predict_df)


#    train_df = df[df['Y'].notna()]:
#        
#        df[df['Y'].notna()] selects all rows from df where the boolean series is True. This creates a new dataframe train_df containing only rows with non-missing Y values. This dataframe will be used to train the regression model.

#    predict_df = df[df['Y'].isna()]:
#       
#        df[df['Y'].isna()] selects all rows from df where the boolean series is True. This creates a new dataframe predict_df containing only rows with missing Y values. This dataframe will be used for predicting and imputing the missing Y values.

#In summary, the code splits the original dataframe into two parts: one for training the model (with complete data) and one for predicting the missing values (with incomplete data).



# In[73]:


# Train the regression model
model = LinearRegression()
#Este código está utilizando la biblioteca de scikit-learn en Python para entrenar un modelo de regresión lineal.

# LinearRegression() es una clase proporcionada por scikit-learn que implementa un modelo de regresión lineal. 
# Este modelo se utiliza para predecir valores continuos basados en una relación lineal entre las características (variables independientes) y la variable objetivo (variable dependiente).

# model es el objeto que se crea como una instancia de la clase LinearRegression(). 
# Este objeto representa nuestro modelo de regresión lineal y contendrá los parámetros aprendidos después de entrenar el modelo.

model.fit(train_df[['X']], train_df['Y'])

#     fit() es un método en scikit-learn que se utiliza para ajustar (entrenar) el modelo a los datos proporcionados. En este caso, se está entrenando el modelo de regresión lineal utilizando los datos en train_df.

#     train_df[['X']] se refiere a la columna 'X' en el DataFrame train_df. 
#     En scikit-learn, las características (variables independientes) 
#     se pasan como un DataFrame o matriz bidimensional.
#     Es importante notar que [['X']] se usa en lugar de ['X'] para asegurar que se pase una estructura de datos 
#     bidimensional, ya que scikit-learn espera que los predictores (X) estén en una matriz 2D.

#     train_df['Y'] se refiere a la variable objetivo (variable dependiente) 
#     que estamos tratando de predecir utilizando el modelo.

# En resumen, este código crea un modelo de regresión lineal y lo entrena utilizando los datos en train_df.
# Una vez entrenado, el modelo estará listo para hacer predicciones sobre nuevos datos basados en la relación 
# lineal que ha aprendido de los datos de entrenamiento.


# In[74]:


# Predict the missing values
predicted_values = model.predict(predict_df[['X']])

#     model.predict() is a method in scikit-learn used to make predictions with a trained model. 
#     In this case, model refers to a previously trained linear regression model.

#     predict_df[['X']] refers to the feature (or input) data from which we want to predict values. 
#     In scikit-learn, input data should typically be passed as a DataFrame or a 2-dimensional array. 
#     Here, [['X']] is used to ensure predict_df['X'] is treated as a DataFrame slice, 
#     ensuring compatibility with scikit-learn's requirements.

#     predicted_values will store the predicted values generated by the model based on the feature data provided 
#     (predict_df[['X']]).

# In essence, after training a linear regression model (model.fit()), this code snippet uses that trained model to predict the values for a new set of data (predict_df). The predictions (predicted_values) will be based on the relationships learned during the model training phase, and these predicted values can be used, 
# for example, to fill in missing values in a dataset or to make forecasts based on new input data.


# In[75]:


# Impute the missing values
df.loc[df['Y'].isna(), 'Y'] = predicted_values

#     df['Y'].isna() checks for missing (NaN) values in the column 'Y' of the DataFrame df. 
#     This condition returns a boolean mask where True indicates that a value is missing.

#     df.loc[condition, 'Y'] accesses the subset of the DataFrame df where the condition (df['Y'].isna()) is True
#     specifically in the column 'Y'. This allows us to target only the rows where 'Y' has missing values.

#     predicted_values is the array or list of values that we obtained from 
#     the previous step where we used a regression model (model.predict(predict_df[['X']])) 
#     to predict missing values.

#     df.loc[df['Y'].isna(), 'Y'] = predicted_values assigns the predicted values (predicted_values) to
#     the subset of df where 'Y' was originally missing. 
#     This effectively fills in the missing values in 'Y' with the values predicted by the model.

# In summary, this code snippet completes the imputation process by replacing missing values in the 'Y' column of df with predicted values obtained from a regression model. This is useful in scenarios where you have missing data that you want to estimate using a predictive model trained on existing data.

print(predicted_values)
print(df.loc[df['Y'].isna(), 'Y'])


# In[76]:


# Display the imputed data
print("\nDataFrame after Imputation:")
print(df)


# In[77]:


#Explanation:

#    Data Preparation: We create a sample dataset with missing values in the Y column.
#    Split Data: We separate the data into two subsets: one for training the regression model (train_df) and one for predicting the missing values (predict_df).
#    Train Model: We fit a linear regression model using the non-missing values.
#    Predict Missing Values: We use the trained model to predict the missing Y values.
#    Impute Missing Values: We replace the missing values in the original dataframe with the predicted values.

