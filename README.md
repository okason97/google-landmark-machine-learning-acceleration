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
![bar plot](https://github.com/okason97/google-landmark-machine-learning-acceleration/blob/master/plots/higher_frequency_landmarks.png)

Datos faltantes
```
# missing data in train data 
print("Missing landmark_id: ",train_data['landmark_id'].value_counts()["None"])
print("Missing url: ",train_data['url'].value_counts()["None"])
```
```
Missing landmark_id:  9260
Missing url:  9260
```

```
# missing data in test data 
print("Missing url: ",test_data['url'].value_counts()["None"])
```
```
Missing url:  2989
```

Landmarks con poca cantidad de imagenes (<1000)
```
# landmark_id with low count
values_count = pd.DataFrame(train_data['landmark_id'].value_counts())
values_count.columns = ["count"]
low_values_count = values_count[values_count["count"] < 1000]
low_values_count
```
------ | count
------ | -----
11073  |	997 
12937  |	988
14703  |	986
12752  |	985
6973   |	977
10313  |	976
4949   |	975
253    |	974
9738   |	971
2665   |	967
10688  |	965
10067  |	960
12075  |	944
7416   |	934
10600  |	934
9135   |	932
1834   |	927
11037  |	922
14111  |	918
13176  |	918
4793   |	911
8161   |	911
3518   |	907
4330   |	907
2330   |	905
6125   |	904
7218   |	897
2802   |	893
5880   |	893
2126   |	888
...    |	...
13002  |	1
6423   |	1
5015   |	1
5030   |	1
14092  |	1
8139   |	1
904    |	1
8309   |	1
1913   |	1
320    |	1
11214  |	1
4047   |	1
2204   |	1
8498   |	1
5414   |	1
3176   |	1
10995  |	1
10420  |	1
11974  |	1
1841   |	1
6261   |	1
9562   |	1
6408   |	1
3230   |	1
10275  |	1
12918  |	1
708    |	1
8797   |	1
9579   |	1
14189  |	1

14835 rows × 1 columns
