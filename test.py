import os
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import array_to_img, load_img, img_to_array
import matplotlib.pyplot as plt

from main import load_train_val_data, path_to_input_image, path_to_target


def show_sample_image(input_img_path, target_path):
    input_img = load_img(input_img_path)
    target_img = load_img(target_path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_img)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(target_img)
    plt.axis("off")
    plt.show()


def display_target(target_array):
    normalized_array = (target_array.astype("uint8")-1) * 127
    plt.imshow(normalized_array[:, :, 0])
    plt.show()


def display_mask(pred, source_img):
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(mask)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(array_to_img(source_img))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    img_size = (200, 200)
    train_input_imgs, train_targets, val_input_imgs, val_targets = load_train_val_data(img_size=img_size)
    # -------------------------------------------------
    RUN_NAME = ""
    os.makedirs(f"Results{RUN_NAME}", exist_ok=True)
    os.chdir(f"Results{RUN_NAME}")
    # -------------------------------------------------

    model = keras.models.load_model("oxford_segmentation.keras")

    num_samples = min(10, len(val_input_imgs))
    random.shuffle(val_input_imgs)
    for i in random.sample(range(len(val_input_imgs)), 10):
        test_img = val_input_imgs[i]
        mask = model.predict(np.expand_dims(test_img, axis=0))[0]
        display_mask(mask, test_img)

