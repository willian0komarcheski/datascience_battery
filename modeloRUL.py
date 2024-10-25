import time
from config import LEARNING_RATE, REGULARIZATION

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import LSTM, GRU, Bidirectional

#arquivo que define o modelo rnn que sera usado

#define as camada rnn
class RNNLayers(layers.Layer):
    def __init__(self, rnn_layers_hidden_node, rnn_layer_type, is_bidirectional=False):
        super(RNNLayers, self).__init__()
        self.rnn_layers = []
        print(rnn_layer_type)

        for hidden_node in rnn_layers_hidden_node:
            if is_bidirectional:
                self.rnn_layers.append(Bidirectional(rnn_layer_type(hidden_node, activation='selu', return_sequences=True,
                                                                    kernel_regularizer=regularizers.l2(REGULARIZATION))))
            else:
                self.rnn_layers.append(rnn_layer_type(hidden_node, activation='selu', return_sequences=True,
                                                      kernel_regularizer=regularizers.l2(REGULARIZATION)))

    def call(self, input_tensor, training):
        x = None
        for layer in self.rnn_layers:
            if x is None:
                x = layer(input_tensor, training=training)
            else:
                x = layer(x, training=training)
        return x

#define as camadas padrões
class DenseLayers(layers.Layer):
    def __init__(self, dense_layer_hidden_nodes):
        super(DenseLayers, self).__init__()
        self.dense_layers = []
        for hidden_node in dense_layer_hidden_nodes:
            self.dense_layers.append(Dense(hidden_node, activation='selu',
                                           kernel_regularizer=regularizers.l2(REGULARIZATION)))

        self.dense_layers.append(Dense(1, activation='linear'))

    def call(self, input_tensor, training):
        x = None
        for layer in self.dense_layers:
            if x is None:
                x = layer(input_tensor, training=training)
            else:
                x = layer(x, training=training)
        return x

#define a estrutura do modelo
class RULModel(Model):
    def __init__(self, input_shape, rnn_layers_hidden_node, dense_layer_hidden_nodes, rnn_layer_type, is_bidirectional=False):
        super(RULModel, self).__init__()
        self.ip_shape = input_shape
        self.rnn = RNNLayers(rnn_layers_hidden_node,
                             rnn_layer_type, is_bidirectional)
        self.dense = DenseLayers(dense_layer_hidden_nodes)

    def call(self, input_tensor, training):
        x = self.rnn(input_tensor, training)
        x = self.dense(x, training)
        return x

    def model(self):
        x = Input(shape=self.ip_shape)
        return Model(inputs=[x], outputs=self.call(x))


#constroi o modelo
def get_model(input_shape, is_bidirectional=False):
    # Model definition
    opt = tf.keras.optimizers.Adam(lr=LEARNING_RATE)

    #define a quantidade de neuronios normais e neuronios rnn
    rnn_layers_hidden_node = [256, 256, 256, 128, 128, 128, 64, 64, 64]
    dense_layer_hidden_nodes = [64, 64, 32, 32]

    #define o tipo de rnn
    rnn_layer_type = LSTM

    #cria um objeto RULModel
    model = RULModel(input_shape, rnn_layers_hidden_node,
                     dense_layer_hidden_nodes, rnn_layer_type, is_bidirectional)
    #roda o modelo
    model.compile(optimizer=opt, loss='huber', metrics=[
                  'mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    #definindo a quantidade de neuronios de entrada de acordo com a quantidade de caracteristica e de acordo com o tamanho da serie temporal
    model.build((1, input_shape[0], input_shape[1]))

    model.summary()
    return model