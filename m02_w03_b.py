# Sus propias funciones:
import my_tf_utils as mytf  # No olvide incluir su función MAKE_BATCHES en su libreria
from my_tf_utils import load_hand_dataset


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops




# Definir la red. No es necesario modificar esta función EXCEPTO para completar la línea de Backpropagation
# Ya conocemos lo que hacen todos estos parámetros, simplemente lo estamos implementando con TF    
data_set = load_hand_dataset()
def model(X_train, Y_train, X_test, Y_test, layers_sizes, eta = 0.0001, N = 1500, batch_size = 32):        
    ops.reset_default_graph()                         # esto es para resetear el modelo
    (X_dims, m) = X_train.shape                       # X_dims: dimensiones de nuestro problema
    Y_dims = Y_train.shape[0]                         # Y_dims: dimensiones de Y (y tamaño de la salida)    
    costs = []                                        
    
    # 1) Crear Placeholders de tamaños [X_dims, None] y [Y_dims, None]    
    #       Usamos 'None' para dejar abierto el número de instancias,
    #       que de hecho van a cambiar en el entrenamiento y prueba
    X, Y = mytf.define_placeholders(X_dims, Y_dims) 
        
    # 2) Inicializa parámetros (pesos W_l y biases b_l)
    params = mytf.init_params(layers_sizes)     
    
    # 3a) Definir Forward propagation y cálculo del costo     
    Z3 = mytf.forward_pass(X, params)

    cost = mytf.compute_cost(Z3, Y)  # Función de costo (no se ha ejecutado, solo se define)
      
    # 3b) Definir el optimizador para backpropagation
    # ---> Backpropagation: debemos definir un optimizador (se ejecuta despues). Usaremos Adam:    
    optimizer = tf.train.AdamOptimizer(learning_rate = eta)   # <-- IMPORTANTE: COMPLETAR ESTA LÍNEA
            
    init = tf.global_variables_initializer() # Inicializa variables
    
    # 4) Iniciamos la sesión 
    with tf.Session() as sess:       
        seed = 3  # para remuestreo de (mini)lotes
        
        # Inicializamos las variables de la sesión
        sess.run(init)        
        # Ciclo de entrenamiento
        for epoch in range(N):  
            epoch_cost = 0.                       # Costo por epoca
            num_batches = int(m / batch_size)     # número de (mini)lotes
            seed = seed + 1                       # para remuestreo de (mini)lotes
            batches = mytf.make_batches(X_train, Y_train, batch_size, seed)  # <-- ocupa su funcion 'make_batches'

            for batch in batches:
                (batch_X, batch_Y) = batch      # Toma un (mini)lote                

                # 5) Ejecutamos los pasos definidos arriba como "optimizer" y "cost"
                #   el diccionario de entrada debe de ser un (mini)lote para X y Y.                
                _ , batch_cost = sess.run([optimizer, cost], feed_dict={X: batch_X, Y: batch_Y})                                
                
                # Acumulamos el costo por época (al final será el promedio):
                epoch_cost += batch_cost / num_batches

            # Reportar y guardar costo
            if epoch % 100 == 0: print ("Costo en epoca %i: %f" % (epoch, epoch_cost))
            if epoch % 5 == 0: costs.append(epoch_cost)
                
        # Graficar el costo
        plt.plot(np.squeeze(costs))
        plt.ylabel('costo')
        plt.xlabel('iteraciones (x 10)')
        plt.title("Tasa de aprendizaje =" + str(eta))
        plt.show()
                
        # 6) Rescatar los parámetros ajustados, para regresarlos
        params = sess.run(params)  
        
        # Desplegar resultados de accuracy
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))   # Calcular aciertos
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # Accuracy
        print ("Accuracy de entrenamiento:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Accuracy de prueba:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return params

    
# --- MAIN (No es necesario modificar) ---
# Cargar dataset
mydata = mytf.load_hand_dataset()   
train_X = mydata["X_train"][:]
test_X = mydata["X_test"][:]
Y_train = mydata["Y_train"][:]
Y_test = mydata["Y_test"][:]
classes = mydata["classes"][:]

# Convertir las etiquetas a codificación "one hot"
train_Y = mytf.as_OneHot(Y_train, len(classes))
test_Y = mytf.as_OneHot(Y_test, len(classes))
# Definir la arquitectura (capas ocultas = ReLU, capa de salida = SoftMax)
layers_sizes = [12288, 25, 12, 6]
params = model(train_X, train_Y, test_X, test_Y, layers_sizes)

