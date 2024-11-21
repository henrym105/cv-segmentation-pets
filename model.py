import tensorflow as tf
import tensorflow.keras as k


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

    outputs = k.layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    model = k.Model(inputs, outputs)
    return model

