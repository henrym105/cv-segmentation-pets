import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import layers, models


def get_model(img_size, num_classes):
    inputs = k.Input(shape=img_size + (3,))
    x = k.layers.Rescaling(1.0 / 255)(inputs)

    x = k.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = k.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = k.layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = k.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = k.layers.Conv2D(256, 3, strides=2, activation="relu", padding="same")(x)
    x = k.layers.Conv2D(256, 3, activation="relu", padding="same")(x)

    x = k.layers.Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
    x = k.layers.Conv2DTranspose(256, 3, strides=2, activation="relu", padding="same")(x)
    x = k.layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = k.layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)
    x = k.layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = k.layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)

    outputs = k.layers.Conv2D(num_classes, 3, activation="linear", padding="same")(x)

    model = k.Model(inputs, outputs)
    return model


def alexnet_model(img_size, num_classes):
    inputs = k.Input(shape=img_size + (3,))

    # 1st Convolutional Layer
    x = k.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu')(inputs)
    x = k.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    
    # 2nd Convolutional Layer
    x = k.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x)
    x = k.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    
    # 3rd Convolutional Layer
    x = k.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    
    # 4th Convolutional Layer
    x = k.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    
    # 5th Convolutional Layer
    x = k.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = k.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    
    # Flatten the layers
    x = k.layers.Flatten()(x)
    
    # 1st Fully Connected Layer
    x = k.layers.Dense(4096, activation='relu')(x)
    x = k.layers.Dropout(0.5)(x)
    
    # 2nd Fully Connected Layer
    x = k.layers.Dense(4096, activation='relu')(x)
    x = k.layers.Dropout(0.5)(x)
    
    # Output Layer
    outputs = k.layers.Dense(num_classes, activation='linear')(x)
    
    model = k.Model(inputs, outputs)
    return model
