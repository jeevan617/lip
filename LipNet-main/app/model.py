import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, BatchNormalization, MaxPooling3D,
    TimeDistributed, Flatten, Dense, Dropout,
    Bidirectional, GRU
)

def build_lipnet_model(input_shape=(75, 50, 100, 1), output_dim=28):
    inputs = Input(name='input', shape=input_shape, dtype='float32')

    x = Conv3D(32, kernel_size=(3, 5, 5), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)

    x = Conv3D(64, kernel_size=(3, 5, 5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)

    x = Conv3D(96, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)

    x = TimeDistributed(Flatten())(x)

    x = Bidirectional(GRU(256, return_sequences=True, dropout=0.5))(x)
    x = Bidirectional(GRU(256, return_sequences=True, dropout=0.5))(x)

    x = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

def load_model(model_type='overlapped'):
    model = build_lipnet_model()
    if model_type == 'overlapped':
        weight_path = os.path.join('models', 'overlapped-weights368.h5')
    elif model_type == 'unseen':
        weight_path = os.path.join('models', 'unseen-weights178.h5')
    else:
        raise ValueError("model_type must be 'overlapped' or 'unseen'")
    model.load_weights(weight_path)
    return model