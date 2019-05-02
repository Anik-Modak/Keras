# The 'Sequential' model is a linear stack of layers.

# Keras Models - Sequential Models
# https://image.slidesharecdn.com/kerasdeeplearning-170829192640/95/deep-learning-using-keras-31-638.jpg?cb=1504119374


# You can create a Sequential model by passing a list of layer instances to the constructor:

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

# OR
# You can also simply add layers via the .add() method:

model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

# Specifying the input shape
'''
 The model needs to know what input shape it should expect. For this reason, 
 the first layer in a Sequential model (and only the first, because following 
 layers can do automatic shape inference) needs to receive information 
 about its input shape. There are several possible ways to do this:

    1. Pass an input_shape argument to the first layer. This is a shape 
       tuple (a tuple of integers or None entries, where None indicates that any 
       positive integer may be expected). In input_shape, the batch dimension is not included.
    
    2. Some 2D layers, such as Dense, support the specification of their input 
       shape via the argument input_dim, and some 3D temporal layers support the arguments input_dim and input_length.
    
    3. If you ever need to specify a fixed batch size for your inputs (this is useful for stateful 
       recurrent networks), you can pass a batch_size argument to a layer. If you pass 
       both batch_size=32 and input_shape=(6, 8) to a layer, it will then expect 
       every batch of inputs to have the batch shape (32, 6, 8). '''

# As the following snippets are strictly equivalent
model = Sequential()
model.add(Dense(32, input_shape=(784,)))

# OR
model = Sequential()
model.add(Dense(32, input_dim=784))

# Compilation
'''
 Before training a model, you need to configure the learning process, which is done 
 via the compile method. It receives three arguments:

    1. An optimizer. This could be the string identifier of an existing optimizer 
       (such as 'rmsprop' or 'adagrad'), or an instance of the Optimizer class. 
       See: https://keras.io/optimizers/

    2. A loss function. This is the objective that the model will try to minimize. 
       It can be the string identifier of an existing loss function (such as 
       categorical_crossentropy or mse), or it can be an objective function.
       See: https://keras.io/losses/
    
    3. A list of metrics. For any classification problem you will want to set 
       this to metrics=['accuracy']. A metric could be the string identifier 
       of an existing metric or a custom metric function.
'''

# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

# For custom metrics
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
