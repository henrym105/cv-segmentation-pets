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


def display_mask(pred, source_img, truth_img, i, save_local=False):
    plt.figure(figsize=(8, 3))

    mask = np.argmax(pred, axis=-1)
    mask = np.expand_dims(mask, axis=-1) * 127
    plt.subplot(1, 3, 1)
    plt.title("Predicted Mask")
    plt.imshow(mask)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Source Image")
    plt.imshow(array_to_img(source_img))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Truth Mask")
    plt.imshow(array_to_img(truth_img))
    plt.axis("off")
    
    if save_local:
        plt.savefig(f"sample_results_{i}.png")
    
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    img_size = (200, 200)
    train_input_imgs, train_targets, val_input_imgs, val_targets = load_train_val_data(img_size=img_size)
    # random.shuffle(val_input_imgs)
    
    # -------------------------------------------------
    RUN_NAME = ""
    os.makedirs(f"Results{RUN_NAME}", exist_ok=True)
    os.chdir(f"Results{RUN_NAME}")
    # -------------------------------------------------
    
    model = keras.models.load_model("pet_segmentation.keras")

    num_samples = 5
    for i in random.sample(range(len(val_input_imgs)), num_samples):
        test_img = val_input_imgs[i]
        test_truth_img = val_targets[i]
        mask = model.predict(np.expand_dims(test_img, axis=0))[0]
        
        display_mask(mask, test_img, test_truth_img, i, save_local=True)

