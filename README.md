# Aceleración de algoritmos de machine learning desde un enfoque arquitectónico

Se desarrollará un estudio detallado sobre el comportamiento de una red neuronal profunda de convolucion (CNN) sobre una arquitectura con GPU utilizando los datos obtenidos de Kaggle de Google Landmark Recognition Challenge.

## Problema a tratar

Se ha seleccionado el dataset utilizado en la competencia de Kaggle Google Landmark Recognition Challenge y se intentara lograr la mayor presicion y el mejor tiempo posible en la prediccion del label utilizando GPU.

### Analisis de datos

Importaremos las librerias necesarias
```
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
```

Luego importaremos los datos
```
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
index = pd.read_csv('../input/index.csv')
```

El tamaño de los datos de entrada y test
```
print("Training data size",train_data.shape)
print("Test data size",test_data.shape)
```
```
Training data size (1225029, 3)
Test data size (117703, 2)
```

Cantidad de landmarks unicos
```
print("Unique landmark_id: ",len(train_data.groupby("landmark_id")["landmark_id"]))
```
```
Unique landmark_id:  14952
```

Los 20 landmarks con mayor numero de imagenes
```
train_data['landmark_id'].value_counts().head(20).plot.bar()
```
(https://github.com/okason97/google-landmark-machine-learning-acceleration/blob/master/plots/higher_frequency_landmarks.png)

