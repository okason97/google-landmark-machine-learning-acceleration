# Aceleración de algoritmos de machine learning desde un enfoque arquitectónico

Se desarrollará un estudio detallado sobre el comportamiento de una red neuronal profunda de convolucion (CNN) sobre una arquitectura con GPU utilizando los datos obtenidos de Kaggle de Google Landmark Recognition Challenge.

## Problema a tratar

Se ha seleccionado el dataset utilizado en la competencia de Kaggle Google Landmark Recognition Challenge y se intentara lograr la mayor presicion y el mejor tiempo posible en la prediccion del label utilizando GPU.

### Analisis de datos

Descargaremos los datos de Kaggle Google Landmark Recognition Challenge https://www.kaggle.com/google/google-landmarks-dataset y los pondremos en la carpeta input.

Importaremos las librerias necesarias:
```
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline 
```

Luego importaremos los datos:
```
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
index = pd.read_csv('../input/index.csv')
```

El tamaño de los datos de entrada y test:
```
print("Training data size",train_data.shape)
print("Test data size",test_data.shape)
```
```
Training data size (1225029, 3)
Test data size (117703, 2)
```
Como se puede ver se posee una gran cantidad de datos de entrenamiento y prueba.

Cantidad de landmarks unicos:
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
![bar_plot](https://github.com/okason97/google-landmark-machine-learning-acceleration/blob/master/plots/higher_frequency_landmarks.png)
El landmark con id 9633 es el que posee la mayor cantidad de imagenes, superando por más de el doble a la cantidad de imagenes del tercer landmark con más imagenes.

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
Hay una gran cantidad de datos faltantes.

Landmarks con poca cantidad de imagenes (<1000)
```
# landmark_id with low count
values_count = pd.DataFrame(train_data['landmark_id'].value_counts())
values_count.columns = ["count"]
low_values_count = values_count[values_count["count"] < 1000]
low_values_count
```
landmark_id | count
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

Cantidad de landmarks con poca cantidad de imagenes (<1000)
```
len(low_values_count)
```
```
14835
```

Landmarks con poca cantidad de imagenes (<100)
```
# landmark_id with low count
values_count = pd.DataFrame(train_data['landmark_id'].value_counts())
values_count.columns = ["count"]
low_values_count = values_count[values_count["count"] < 100]
len(low_values_count)
```
```
12928
```

Landmarks con poca cantidad de imagenes (<10)
```
# landmark_id with low count
values_count = pd.DataFrame(train_data['landmark_id'].value_counts())
values_count.columns = ["count"]
low_values_count = values_count[values_count["count"] < 10]
len(low_values_count)
```
```
6190
```
Como se puede ver, existen una gran cantidad de landmarks con poca cantidad de imagenes, lo que podria dificultar su prediccion.

Landmarks con mucha cantidad de imagenes (>=1000)
```
high_values_count = values_count[values_count["count"] >= 1000]
high_values_count
```

landmark_id |	count
----- | -----
9633 |	50010
6051 |	49806
6599 |	23218
9779 |	18309
2061 |	13183
5554 |	11033
6651 |	9444
None |	9260
5376 |	9161
6696 |	9161
2743 |	8950
4352 |	8928
13526 | 	8617
1553 |	7754
10900 |	6961
8063 |	6612
8429 |	6398
4987 |	5319
12220 | 	5282
11784 |	5221
2949 |	4879
12718 |	3772
3804 |	3669
10184 |	3592
7092 |	3517
10045 |	3426
2338 |	3400
12172 |	3348
3924 |	3347
428 |	3153
... |	...
1546 |	1183
4644 |	1176
9605 |	1166
7420 |	1158
4954 |	1150
6597 |	1147
5475 |	1136
3426 |	1135
3065 |	1112
1310 |	1110
5166 |	1095
5421 |	1093
11536 |	1089
12966 |	1085
6347 |	1083
12647 |	1079
2145 |	1074
9296 |	1073
10644 |	1065
10005 |	1064
7840 |	1049
12965 |	1049
3034 |	1044
6846 |	1043
10496 |	1040
5618 |	1037
13471 |	1036
5206 |	1023
7130 |	1007
2246 |	1006

Cantidad de landmarks con mucha cantidad de imagenes (>=1000)
```
len(high_values_count)
```
```
117
```

Veamos algunas imagenes

```
from IPython.display import Image
from IPython.core.display import HTML 

def display_category(urls, category_name):
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(12).iteritems()])

    display(HTML(images_list))
    
category = train_data['landmark_id'].value_counts().keys()[0]
urls = train_data[train_data['landmark_id'] == category]['url']
display_category(urls, "")
```
![some_images](https://github.com/okason97/google-landmark-machine-learning-acceleration/blob/master/plots/someimages.png)
Se puede ver que las imagenes poseen diferente iluminación, diferente tamaño y vistas a landmarks obstaculizadas.

### Preparación de datos

Dada la gran cantidad de datos, comenzaremos prediciendo los primeros 10 landmarks con mayor cantidad de imagenes asociadas.

landmark_id |	#urls
----- | -----
9633 |	50010
6051 |	49806
6599 |	23218
9779 |	18309
2061 |	13183
5554 |	11033
6651 |	9444
5376 |	9161
6696 |	9161
2743 |	8950

Dados los diferentes tamaños de las imagenes, estas deberan ser modificadas para poseer todas el mismo tamaño (256x256). En caso de que el ratio de la imagen difiera demasiado del buscado, se optara por recortarla.

Se reduciran dimensionalidad de las imagenes cambiando de canales RGB a un solo canal en escala de grises.

Luego, se realizara ecualizacion de histograma para aumentar el contraste en las imagenes.

Finalmente, se normalizaran las imagenes, esto permite llegar más rapido a la convergencia. Se resta la media a cada pixel y se divide por la desviacion estandar, luego se escalan los resultados al rango [0,255].

### Definición del modelo

Se utilizara una CNN en tensorflow. Con 
Hiperparametros
batch_size = 256

## Fuentes
https://becominghuman.ai/image-data-pre-processing-for-neural-networks-498289068258
https://benanne.github.io/2015/03/17/plankton.html#prepro-augmentation
