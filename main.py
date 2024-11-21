import os
import matplotlib.pyplot as plt
import numpy as np
import random
import json

import tensorflow as tf
from tensorflow.keras.models import load_model, load_img, img_to_array
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from model import get_model


def path_to_input_image(path, img_size):
    return img_to_array(load_img(path, target_size=img_size))


def path_to_target(path, img_size):
    img = img_to_array(load_img(path, target_size=img_size, color_mode="grayscale"))
    img = img.astype("uint8") - 1
    return img


def load_train_val_data(
        input_dir = "images/", 
        target_dir = "annotations/trimaps/", 
        img_size = (200,200), 
        num_val_samples = 1000
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load images into memory.

    Args:
        input_dir: str, path to the input images
        target_dir: str, path to the target images
        img_size: tuple, size of the images
        num_val_samples: int, number of validation samples

    Returns:
        tuple: train_input_imgs, train_targets, val_input_imgs, val_targets
    """
    input_img_paths = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".jpg")])
    target_paths = sorted([os.path.join(target_dir, fname) for fname in os.listdir(target_dir) if fname.endswith(".png") and not fname.startswith(".")])

    # Set image size and number of images
    num_imgs = len(input_img_paths)

    # Shuffle the image paths
    random.Random(1).shuffle(input_img_paths)
    random.Random(1).shuffle(target_paths)

    # Initialize arrays to hold the images and targets
    input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")
    targets = np.zeros((num_imgs,) + img_size + (1,), dtype="uint8")
    
    # Load images and targets into arrays
    for i in range(num_imgs):
        input_imgs[i] = path_to_input_image(input_img_paths[i], img_size)
        targets[i] = path_to_target(target_paths[i], img_size)

    # Split data into training and validation sets
    train_input_imgs = input_imgs[:-num_val_samples]
    train_targets = targets[:-num_val_samples]
    val_input_imgs = input_imgs[-num_val_samples:]
    val_targets = targets[-num_val_samples:]

    return train_input_imgs, train_targets, val_input_imgs, val_targets


def save_history_plot(history):
    # Plot training and validation loss
    epochs = range(1, len(history.history["loss"]) + 1)
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig("training_validation_loss.png")
    plt.close()


if __name__ == "__main__":
    # Set image size and number of images
    print("Physical GPUs:", tf.config.list_physical_devices('GPU'))
    img_size = (200, 200)
    train_input_imgs, train_targets, val_input_imgs, val_targets = load_train_val_data(img_size = img_size)

    # -------------------------------------------------
    RUN_NAME = ""
    # -------------------------------------------------
    os.makedirs(f"Results{RUN_NAME}", exist_ok=True)
    os.chdir(f"Results{RUN_NAME}")

    # Get the model and print its summary
    if os.path.exists("oxford_segmentation.keras"):
        model = load_model("oxford_segmentation.keras")
    else:
        model = get_model(img_size=img_size, num_classes=3)
    model.summary()

    # Compile the model
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

    # Define callbacks
    callbacks = [
        ModelCheckpoint("oxford_segmentation.keras", save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
    ]

    # Train the model
    params = {
        "epochs": 10,
        "batch_size": 64,
    }
    with open("training_params.json", "w") as json_file:
        json.dump(params, json_file, indent=4)

    history = model.fit(
        train_input_imgs, 
        train_targets, 
        validation_data=(val_input_imgs, val_targets), 
        callbacks=callbacks,
        **params
    )

    # Save training history to a file
    with open(f"training_history.npy", "wb") as f:
        np.save(f, history.history)
        
    save_history_plot(history)
