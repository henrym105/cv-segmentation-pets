# cv-segmentation-pets

This repository contains code for training and evaluating a convolutional neural network (CNN) for image segmentation on the Oxford-IIIT Pet Dataset. The dataset consists of images of 37 different breeds of cats and dogs, with pixel-level annotations for segmentation.

## Setup

### Prerequisites

- Python 3.11.0
- see more in `requirements.txt`

### Sample Results

Below are some sample results from the model:

![Sample Result 1](Results/sample_results_271.png)
![Sample Result 2](Results/sample_results_874.png)


# Model Summary

Below is the summary of the Tensorflow model architecture:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)             │ (None, 200, 200, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ rescaling (Rescaling)                │ (None, 200, 200, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d (Conv2D)                      │ (None, 100, 100, 64)        │           1,792 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 100, 100, 64)        │          36,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 50, 50, 128)         │          73,856 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_3 (Conv2D)                    │ (None, 50, 50, 128)         │         147,584 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_4 (Conv2D)                    │ (None, 25, 25, 256)         │         295,168 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_5 (Conv2D)                    │ (None, 25, 25, 256)         │         590,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_transpose (Conv2DTranspose)   │ (None, 25, 25, 256)         │         590,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_transpose_1 (Conv2DTranspose) │ (None, 50, 50, 256)         │         590,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_transpose_2 (Conv2DTranspose) │ (None, 50, 50, 128)         │         295,040 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_transpose_3 (Conv2DTranspose) │ (None, 100, 100, 128)       │         147,584 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_transpose_4 (Conv2DTranspose) │ (None, 100, 100, 64)        │          73,792 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_transpose_5 (Conv2DTranspose) │ (None, 200, 200, 64)        │          36,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_6 (Conv2D)                    │ (None, 200, 200, 3)         │           1,731 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
Total params: 5,761,288 (21.98 MB)
Trainable params: 2,880,643 (10.99 MB)
Non-trainable params: 0 (0.00 B)
Optimizer params: 2,880,645 (10.99 MB)
```


### Installation

1. Clone the repository:
```sh
git clone https://github.com/henrym105/cv-segmentation-pets.git
cd cv-segmentation-pets
```

2. Install the required packages:
```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Download the Oxford-IIIT Pet Dataset and place the images in the `images/` directory and the annotations in the `annotations/trimaps/` directory.

## Usage

### Training

To train the model, run the following command:
```sh
python main.py
```

### Testing

To test the model, run the following command:
```sh
python test.py
```

# Dataset

To download the dataset: 
```sh
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xf images.tar.gz
tar -xf annotations.tar.gz
```

The Oxford-IIIT Pet Dataset is a 37-category pet dataset with roughly 200 images for each class. The images have large variations in scale, pose, and lighting. All images have associated ground truth annotations of breed, head ROI, and pixel-level trimap segmentation.

### Contents

- `trimaps/`: Trimap annotations for every image in the dataset. Pixel Annotations: 1: Foreground, 2: Background, 3: Not classified.
- `xmls/`: Head bounding box annotations in PASCAL VOC Format.
- `list.txt`: Combined list of all images in the dataset. Each entry in the file is of the following nature: `Image CLASS-ID SPECIES BREED ID`.
- `trainval.txt`: Files describing splits used in the paper.

## Model

The model architecture is defined in `model.py`. It uses a convolutional neural network (CNN) for image segmentation. The model can be trained from scratch or loaded from a pre-trained checkpoint.


## Results

Training and validation loss plots are saved as `training_validation_loss.png` in the results directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.