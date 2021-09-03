# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%

# Aqui estan todos los imports que se han usado en este ejemplo.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap


#%%

class Perceptron(object):
    """
    @Parametros
        lr : float
            Learning rate (Ritmo de aprendizaje). Su valor debe de estar en el rango del 0.0 al 1.0
            Cuanto mas cerca este de 0.0, mas tardara el programa.
        iteraciones: int
            Numero de iteraciones que se realizaran sobre el banco de datos de entrenamiento.
            A este parametro se le conoce como epochs (epocas). En los siguientes ejemplos,
            usare esta nomenclatura.
        semilla: int 
            Numero entero que se usara para crear el generador de numeros aleatorios que obtendra
            los pesos iniciales del perceptron.
    """
    
    def __init__(self ,lr=0.01, iteraciones=25, semilla=1):
        self.lr = lr
        self.iteraciones = iteraciones
        self.semilla = semilla
    
    def predict(self, X_training_features):
        """
        Funcion auxiliar. Se usa en la funcion fit. Aplica la funcion auxiliar calculo_de_la_red()
        a los elementos que cumplan la condicion.
        """
        return np.where(self.calculo_de_la_red(X_training_features) >= 0.0, 1, -1)
    
    def calculo_de_la_red(self, X_training_features):
        """
        Multiplica las caracteristicas de la red por los pesos del 1 a N.
        """
        return np.dot(X_training_features, self.pesos[1:]) + self.pesos[0]
    
    def fit(self, X_training_features, Y_training_labels):
        """
        Esta funcion entrena el perceptron.
        @Parametros
            X_training_features : Matriz. Forma = [Numero_de_ejemplos, numero_de_caracteristicas]
                Contiene los datos ya tratados que se usaran para entrenar el modelo.
                Ejemplo:
                    Especie color_del_tallo altura_del_tallo numero_de_hojas
                 
                 1º   0           1                 0.4             0.1
                 2º   0           1                 0.3             0.12
                 3º   1           2                 0.6             0.134
                 4º   1           2                 0.2             0.893
                 5º   0           1                 0.4             0.343
            
            Y_training_labels : Vector. Forma = [Numero_de_ejemplos]
                Etiquetas que clasifican las entradas del conjunto X_training_features.
        """
        
        #Creamos el genrador de numero aleatorios
        generador_aleatorio = np.random.RandomState(self.semilla)
        
        #Obtengo el tamaño que tendra el vector de pesos
        size_pesos = 1 + X_training_features.shape[1]
        
        #Obtengo el array inicial de pesos(w) a partir de una distribucion normal. Una distribucion
        #normal es aquella que solo genera numero entre el 0 y el 1, a grandes rasgos.
        self.pesos = generador_aleatorio.normal(loc=0.0, scale=0.01, size=size_pesos)
    
        # Realizo el numero de iteraciones indicado sobre el conjunto de datos.
        for _ in range(self.iteraciones):
            
            # La funcion zip convierte junta ambas estructuras en una sola. De esta forma podemos
            # iterar con un simple for tantos los datos de entrenamientos(X_training_features) como
            # las etiquetas (Y_training_labels)
            for x, y in zip(X_training_features, Y_training_labels):
                
                # Aprendemos (Que es basicamente multiplicar para poder actualizar el vector de pesos)
                learn = self.lr * (y - self.predict(x)) 
                
                # Actualizamos los pesos
                self.pesos[1:] += learn * x
                
                self.pesos[0] += learn
            
        return self
            
#%%

####################################
# SECCION 0: OBTENIENDO LOS DATOS
####################################

# Nos descargamos la base de datos y la leemos con pandas.

iris_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(iris_path, header=None, encoding='utf-8')

print(df.tail())

#%%

####################################
# SECCION 1: VISUALIZANDO LOS DATOS
####################################

"""
Vamos a crear un grafico scatter para ver como estan repartidos los datos en funcion de dos de
sus caracteristicas (features). En este caso, vamos a usar la longuitud del sepalo y la longuitud
del petalo. (sepal length and petal length.)
"""


# Obtenemos los valores de la cuarta columna de la base de datos. Esta columna contiene la especie
# a la que pertenece la flor.
y = df.iloc[0:100, 4].values

# Cambiamos la especie a la que pertenece la flor por -1 o 1. Si es 'Iris-setosa' la
# clasificamos como -1. En caso contrario, la clasificamos como 1.
y = np.where(y == 'Iris-setosa', -1, 1)

# Obtenemos una matriz 100x2 que contiene las dos primeras caracteristicas de las flores.
# Estas caracteristicas son: sepal length y petal length.
X = df.iloc[0:100, [0,2]].values

"""
Aqui comenzamos a usar matplotlib. Al llamar por primera vez a plt.scatter, matplotlib le asignara
todas las llamadas que hagamos a la biblioteca a esa grafico hasta que llamemos a la funcion
plt.show(). Esto significa, que podemos alterar el grafico en varias lineas.

En la primera y la segunda linea, añadimos los puntos al grafico. Para esto hay que pasarle
los valores x,y que vamos ha usar. De 0 a 49 las flores son setosa y de 50 a 100 son versicolor.
"""

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')

plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

# Añadimos el nombre del eje X
plt.xlabel('sepal length [cm]')

# Añadimos el nombre del eje Y
plt.ylabel('petal length [cm]')

# Añadimos una leyenda a la parte superior izquierda del grafico
plt.legend(loc='upper left')

# Creamos el plot
plt.show()

#%%

####################################
# SECCION 2: ENTRENANDO EL MODELO
####################################

"""
Aqui la magia ocurre dentro de la clase Perceptron. Os animo a aumentar o disminuir los parametros
learning rate (lr) y las iteraciones. Obtendreis resultados distintos, algunos muy malos.

Por ejemplo: lr =0.0001 , iteraciones=5

Encontrar los parametros perfectos para un problema en concreto es dificil. Hay metodos para hacerlo
que veremos mas adelante en el curso. A mi, una de mas partes que mas me gusta de este campo es
sentarme durante horas cambiando parametros para ver como reacciona el modelo.

¿Un uso del tiempo infeciente? Quizas...
"""

ppr = Perceptron(lr=0.1, iteraciones=15)

ppr.fit(X,y)



#%%

####################################
# SECCION 3: CLASIFICANDO LOS DATOS
####################################

def plot_decision_regions(X, y, classifier, resolution=0.01):
    """
    Aqui vamos a ver como se realiza el grafico azul-rojo.
    
    Hemos usado los mismos datos del grafico anterior, pero esta vez hemos creado una linea recta
    usando los pesos del perceptron. Esta linea separa los dos conjuntos a la perfeccion, si no la
    has liado cambiando los parametro iniciales del Perceptron.
    """
    
    # Al utilizar markers, podemos cambiar el como se ven los puntos en el plot.
    # La 's' se trasnforma en un cuadrado. Puedes ver todas las posibilidades en
    # esta pagina web: https://matplotlib.org/stable/api/markers_api.html
    markers = ('s','x')
    
    # Los colores que vamos ha usar en el plot. Quien lo iba a decir eh?
    colors = ('red','blue')
    
    # Creamos el cmap en funcion del numero de etiquetas. Si le añadimos mas colores y marcas
    # a las variables de arriba, podemos tratar con mas caracteristicas. Lo veremos mas adelante.
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Obtenemos los minimos y maximos de las caracteristicas que hemos usado. Le sumamos/restamos
    # +-0.5 para dejar mas espacio en el grafico y que se vea mejor.
    x1_min, x1_max = X[:, 0].min() -0.5 , X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() -0.5 , X[:, 1].max() + 0.5
    
    # Obtenemos dos vectores de coordenadas entre el minimo y el maximo de cada caracteristica
    # (feature)
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    
    # Clasificamos los datos y obtenemos un vector con las etiquetas.
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    # Dibujamos la linea utilizando las coordenadas de xx1 y xx2.
    plt.contourf(xx1,xx2,Z,alpha=0.3,cmap=cmap)
    
    # Fijamos los limites minimos y maximos de los ejes cardinales del plot.
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    
    # Creamos tantos plots como etiquetas unicas halla. En este caso 2.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha = 0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

# Llamamos a la funcion.
plot_decision_regions(X,y,classifier=ppr)

# Añadimos los nombres de la coordenadas x,y a el plot
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')

# Añadimos la leyenda.
plt.legend(loc='upper left')

# Monstramos el esquema.
plt.show()



#%%