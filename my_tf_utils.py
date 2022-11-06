import h5py
import numpy as np
import tensorflow as tf






def define_placeholders(n_x, n_y):    
    # tf functions to use:
    #   tf.placeholder
    #   tf.float32   (wrapper for Dtype float)
    X = tf.placeholder(tf.float32, [n_x, None])
    Y = tf.placeholder(tf.float32, [n_y, None])

    # (2 líneas de código)
    return X, Y

def init_params(layersSizes=[12288, 25, 12, 6]):
#   Returns: Diccionario con los parámetros correspondientes: W1, b1, W2, b2, etc.
    # Funciones a utilizar:
    #   tf.get_variables
    params = {}
    for idx in range(1, len(layersSizes)):
        params[f"W{idx}"] = tf.compat.v1.get_variable(name=f"W{idx}",
                                                    shape=[layersSizes[idx],
                                                    layersSizes[idx-1]],
                                                    initializer=tf.compat.v1.initializers.glorot_normal) 
        params[f"b{idx}"] = tf.compat.v1.get_variable(name=f"b{idx}",
                                                    shape=[layersSizes[idx], 1],
                                                    initializer=tf.compat.v1.initializers.zeros)
    #   tf.contrib.layers.xavier_initializer = tf.compat.v1.initializers.glorot_normal
    #   tf.zeros_initializer = tf.compat.v1.initializers.zeros
    # Documentation https://www.tensorflow.org/api_docs/python/tf/compat/v1/initializers
    # (4 líneas de código)
    return params
    

def forward_pass(X, params):
    # Entradas:
    #    X -- placeholder para datos de entrada con tamaño: (dimensiones, instancias)
    #    params -- diccionario con los parámetros de la red
    # Regresa: 
    #   Z3 -- la salida de la tercera capa >> ANTES << de la función de activación 
    #         porque la función que calcula el costo aplicará la última función de activación.
    # Notas:
    #   La activación para estas capas es ReLU
    
    
    A = tf.cast(X, tf.float32)
    
    for idx in range(1, int(len(params)/2)):
        print("Z.shape")
        
        Z = tf.add(tf.matmul(params[f"W{idx}"], A), params[f"b{idx}"]) 
        print(Z.shape)
        A = tf.nn.relu(Z)
    Z3 = tf.add(tf.matmul(params[f"W{idx+1}"], A), params[f"b{idx+1}"])
    A3 = tf.nn.softmax(Z3)

    # Z1 = tf.add(tf.matmul(params["W1"], tf.cast(X, tf.float32)), params["b1"])
    # A1 = tf.nn.relu(Z1)
    
    # Z2 = tf.add(tf.matmul(params["W2"], A1), params["b2"])
    # A2 = tf.nn.relu(Z2)

    # Z3 = tf.add(tf.matmul(params["W3"], A2), params["b3"])
    # A3 = tf.nn.softmax(Z3)  
        
    # tf.nn.relu(, name=None)
    #   Usar funciones de TF, no de numpy!
    # (aprox. 10 líneas de código)    
    return A3

def compute_cost(Z3, Y):
    # Entradas: 
    #   Z3 -- salida de 'forward_pass'
    #   Y -- es un Placeholder para las etiquetas "verdaderas", con el mismo tamaño que Z3    # 
    # Regresa: 
    #   cost -- el costo    #
    # Funciones a utilizar:
    #   tf.transpose <- importante para que las variables tengan la forma requerida
    #   tf.nn.softmax_cross_entropy_with_logits
    
    input_tensor = tf.nn.softmax_cross_entropy_with_logits(
                            labels=tf.transpose(tf.cast(Y, tf.float32)), logits=Z3
                        )
    cost = tf.math.reduce_mean(tf.math.reduce_mean(input_tensor=input_tensor))
    #   tf.reduce_mean
    # (3 líneas de código)    
    return cost

def as_OneHot(Y, C=6):        
    # (1 elegante línea de código)
    # return tf.reshape(tf.one_hot(indices=Y, depth=C), shape=(C, -1)) # Pues no es elegante, pero se hizo lo que se pudo.
    N = np.zeros((Y.T.size, Y.T.max()+1))
    N[np.arange(Y.T.size),Y.T] = 1
    return N

def make_batches(X, Y, batch_size=64, seed=0):
    np.random.seed(seed) # Para tener resultados reproducibles
    # Y.numpy()
    shuffler = np.random.permutation(len(Y.T))

    X = X.T[shuffler]
    X = X.T
    Y = Y.T[shuffler]   
    Y = Y.T


    btch = len(X.T) // batch_size
    batches = []
    for b in range(btch):
        x = X.T[b*batch_size:(b+1)*batch_size]
        y = Y.T[b*batch_size:(b+1)*batch_size]
        batches.append((x.T, y.T))
    
    return batches
    
# def make_batches(X, Y, batch_size=64, seed=0):
#     np.random.seed(seed) # Para tener resultados reproducibles
    
#     # shuffler = np.random.permutation(Y.shape[-1])

#     X = tf.random.shuffle(
#                 value=X.T, seed=seed
#             )
#     X = X.T
#     Y = tf.random.shuffle(
#                 value=Y.T, seed=seed
#             )
#     Y = Y.T


#     btch = X.shape[-1] // batch_size
#     batches = []
#     for b in range(btch):
#         x = X.T[b*batch_size:(b+1)*batch_size]
#         y = Y.T[b*batch_size:(b+1)*batch_size]
#         batches.append((np.array(x.T), np.array(y.T)))
    
#     return batches
    

# funcion para cargar datos. No necesita modificar esta funcion 
def load_hand_dataset():
    train = h5py.File('manos_train.h5',"r")
    X_train = np.array(train["train_set_x"][:]) 
    Y_train = np.array(train["train_set_y"][:]) 
    test = h5py.File('manos_test.h5',"r")
    X_test = np.array(test["test_set_x"][:])
    Y_test = np.array(test["test_set_y"][:])
    classes = np.array(test["list_classes"][:])    
    Y_train = Y_train.reshape((1, Y_train.shape[0]))
    Y_test = Y_test.reshape((1, Y_test.shape[0]))
    X_train_flat = X_train.reshape(X_train.shape[0], -1).T
    X_test_flat = X_test.reshape(X_test.shape[0], -1).T    
    X_train = X_train_flat/255.
    X_test = X_test_flat/255.
    d = {"X_train": X_train,
         "Y_train": Y_train, 
         "X_test" : X_test, 
         "Y_test" : Y_test, 
         "classes" : classes}
    return d

# d = load_hand_dataset()
# X_train = d['X_train']
# X_test = d['X_test']
# Y_train = d['Y_train']
# Y_test = d['Y_test']
# train_Y = as_OneHot(Y=Y_train)
# params = init_params()
# A3 = forward_pass(X=X_train, params=params)
# # tf.compat.v1.reset_default_graph()
# init = tf.compat.v1.global_variables_initializer()
# cost = compute_cost(Z3=A3, Y=train_Y)
# with tf.compat.v1.Session() as sess:
#     sess.run(init)
#     c = sess.run(cost)
#     print(c)
    
    

