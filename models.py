from tensorflow.keras import Model, initializers, activations
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D
import tensorflow.keras.layers as l
import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True) # 可以调试layer和model里的call，不用graphic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import backend as K
from config import *


class CBN(l.Layer):
    def __init__(self, filters, kernel_size, conv_strides, pool_size, pool_strides, conv_padding='valid',
                 pool_padding='same', se=False, **kwargs):
        super().__init__(**kwargs)
        self.l_Conv2D = Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=conv_strides,
                               bias_initializer=initializers.Constant(value=0.1),
                               padding=conv_padding)
        self.l_bn = BatchNormalization()
        self.l_relu = ReLU()
        self.l_mp = MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding=pool_padding)

        self.r = 16  # 之前是4
        self.gap = l.GlobalAveragePooling2D()
        self.d1 = l.Dense(filters // self.r, use_bias=False, activation='relu')
        self.d2 = l.Dense(filters, use_bias=False, activation='sigmoid')

        self.se = se

    def call(self, inputs, **kwargs):
        x = self.l_Conv2D(inputs)
        x = self.l_bn(x)
        x = self.l_relu(x)
        x = self.l_mp(x)
        # senet
        if self.se:
            s = self.gap(x)
            s = self.d1(s)
            s = self.d2(s)
            x = l.Multiply()([x, s])
        return x

    def get_config(self):
        config = super().get_config().copy()
        # config.update({
        #
        # })
        return config


def reco_model_cons(n_classes, phase_input_shape, magn_input_shape):
    phase_input = l.Input(shape=phase_input_shape)
    magn_input = l.Input(shape=magn_input_shape)
    # phase
    x_p = CBN(8, kernel_size=(3, 8), conv_strides=(1, 1), pool_size=(1, 3), pool_strides=(1, 3))(phase_input)
    x_p = CBN(16, kernel_size=(3, 8), conv_strides=(1, 1), pool_size=(1, 4), pool_strides=(1, 4))(x_p)
    x_p = CBN(32, kernel_size=(3, 5), conv_strides=(1, 1), pool_size=(2, 3), pool_strides=(2, 3))(x_p)
    x_p = CBN(32, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(1, 3), pool_strides=(1, 3))(x_p)
    x_p = CBN(32, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(2, 2), pool_strides=(2, 2))(x_p)
    x_p = l.Flatten()(x_p)
    x_p = l.Dense(256, activation='relu', bias_initializer=initializers.Constant(value=0.1))(x_p)
    # magn
    x_m = CBN(8, kernel_size=(3, 8), conv_strides=(1, 1), pool_size=(1, 3), pool_strides=(1, 3))(magn_input)
    x_m = CBN(16, kernel_size=(3, 8), conv_strides=(1, 1), pool_size=(1, 4), pool_strides=(1, 4))(x_m)
    x_m = CBN(32, kernel_size=(3, 5), conv_strides=(1, 1), pool_size=(2, 3), pool_strides=(2, 3))(x_m)
    x_m = CBN(32, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(1, 3), pool_strides=(1, 3))(x_m)
    x_m = CBN(32, kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(2, 2), pool_strides=(2, 2))(x_m)
    x_m = l.Flatten()(x_m)
    x_m = l.Dense(256, activation='relu', bias_initializer=initializers.Constant(value=0.1))(x_m)
    # concat
    fusion_embedding = l.concatenate([x_p, x_m])
    flattened_fe = l.Flatten()(fusion_embedding)

    flattened_fe = l.Dense(128, activation='relu', bias_initializer=initializers.Constant(value=0.1))(flattened_fe)

    dropout_fe = l.Dropout(0.5)(flattened_fe)
    output = l.Dense(n_classes, activation=activations.get('softmax'))(dropout_fe)
    return Model(inputs=[phase_input, magn_input], outputs=[output])


def cons_cnn_model_gai(input_shape):
    cnn_model = Sequential(name='5_layer_CNN')
    cnn_model.add(Conv2D(8,
                         kernel_size=(3, 8),
                         strides=(1, 1),
                         input_shape=input_shape,
                         bias_initializer=initializers.Constant(value=0.1),
                         name='Conv_1'))
    cnn_model.add(BatchNormalization(name='BN_1'))
    cnn_model.add(ReLU(name='Relu_1'))
    cnn_model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same', name='MP_1'))

    cnn_model.add(Conv2D(16,
                         kernel_size=(3, 8),
                         strides=(1, 1),
                         bias_initializer=initializers.Constant(value=0.1),
                         name='Conv_2'))
    cnn_model.add(BatchNormalization(name='BN_2'))
    cnn_model.add(ReLU(name='Relu_2'))
    cnn_model.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same', name='MP_2'))

    cnn_model.add(Conv2D(32,
                         kernel_size=(3, 5),
                         strides=(1, 1),
                         bias_initializer=initializers.Constant(value=0.1),
                         name='Conv_3'))
    cnn_model.add(BatchNormalization(name='BN_3'))
    cnn_model.add(ReLU(name='Relu_3'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 3), strides=(2, 3), padding='same', name='MP_3'))

    cnn_model.add(Conv2D(32,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         bias_initializer=initializers.Constant(value=0.1),
                         name='Conv_4'))
    cnn_model.add(BatchNormalization(name='BN_4'))
    cnn_model.add(ReLU(name='Relu_4'))
    cnn_model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same', name='MP_4'))

    cnn_model.add(Conv2D(32,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         bias_initializer=initializers.Constant(value=0.1),
                         name='Conv_5'))
    cnn_model.add(BatchNormalization(name='BN_5'))
    cnn_model.add(ReLU(name='Relu_5'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='MP_5'))

    cnn_model.add(Flatten(name='Flatten'))
    cnn_model.add(Dense(128, bias_initializer=initializers.Constant(value=0.1), name='Dense_1'))
    return cnn_model


def au_model_cons():
    phase_input_shape = (NUM_OF_FREQ * N_CHANNELS, PADDING_LEN, 1)
    input_shape = phase_input_shape
    backbone = cons_cnn_model_gai(input_shape)
    l2_norm = l.Lambda(lambda embeddings: K.l2_normalize(embeddings, axis=1), name='l2_norm')
    input = l.Input(shape=input_shape)
    embeddings = backbone(input)
    embeddings = l2_norm(embeddings)
    model = Model(inputs=input, outputs=embeddings)
    return model
