import glob
import os
from os.path import join
from pathlib import Path
from typing import List, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preproc_inception
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preproc_mobile
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preproc_resnet
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.StretchOutNormalize import StretchOutNormalize

# define variables used throughout script
ROOT: Path = Path(__file__).resolve().parent
IMG_CHANNELS: int = 3
IMG_DIMS: Tuple[int, int] = (165, 165)
IMG_SIZE_FULL: Tuple[int, int, int] = IMG_DIMS + (IMG_CHANNELS,)

"""
THIS BLOCK WAS USED TO PREP DATA FOR NETWORKS
No use since I have previously performed the extraction and cleaning
I left it in as it is to provide the option to run it if necessary
from src.DataPrep import DataPrep

# data preparation using DataPrep class and methods
prepared_data: DataPrep = DataPrep(10)
prepared_data.extract_data(save_dir=str(join(ROOT, "data/")))
# spilt train and test
prepared_data.food_directory_split(
    path="data/food-101/meta/train.txt", source="data/food-101/images", destination="data/food-101/train"
)
prepared_data.food_directory_split(
    path="data/food-101/meta/test.txt", source="data/food-101/images", destination="data/food-101/test"
)
# resize images to be used for training networks
prepared_data.resize(path="data/food-101/train/")
prepared_data.resize(path="data/food-101/test/")
"""


# Model training
def inception_model(
    generator_img_shape: Tuple[int, int],
    inception_input_shape: Tuple[int, int, int],
    training_dir: str,
    testing_dir: str,
    save_dir: str,
    batch_size: int = 4,
    epochs: int = 5,
    num_classes: int = 2,
) -> Model:
    """
    This function performs transfer learning on the InceptionV3 model, initially trained on "imagenet" data
    :param generator_img_shape: Length and width of the images
    :param inception_input_shape: Length width and channels of images (channels=1 for greyscale, 3 for colour)
    :param training_dir: directory of the training data
    :param testing_dir: directory of the testing data
    :param save_dir: Where to save the model
    :param batch_size: size of the batch
    :param epochs: number of epochs to train
    :param num_classes: number of classes we are training
    :return: Trained model
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        model_path: str = save_dir
    else:
        model_path = save_dir

    # using image aurgmentation/preprocessing https://keras.io/api/preprocessing/image/
    data_gen_train: ImageDataGenerator = ImageDataGenerator(preprocessing_function=preproc_inception)
    # generate batches of augmented data
    gen_train: ImageDataGenerator = data_gen_train.flow_from_directory(
        training_dir, target_size=generator_img_shape, batch_size=batch_size, class_mode="categorical"
    )
    gen_test: ImageDataGenerator = data_gen_train.flow_from_directory(
        testing_dir, target_size=generator_img_shape, batch_size=batch_size, class_mode="categorical"
    )
    # assign inceptionv3 model, do not include final layers, we will create those ourselves
    inception_V3: Model = InceptionV3(
        weights="imagenet", input_shape=inception_input_shape, include_top=False, pooling=None
    )
    # inception_V3.summary()
    output = inception_V3.output
    # add pooling layer
    max_pool_output: MaxPool2D = MaxPool2D()(output)
    flatten_output: Flatten = Flatten()(max_pool_output)
    dense_output: Dense = Dense(128, activation="relu")(flatten_output)
    drop_output: Dropout = Dropout(0.1)(dense_output)
    # make prediction using soft max
    pred: Dense = Dense(num_classes, kernel_regularizer=regularizers.l2(0.01), activation="softmax")(drop_output)
    # define, compile, and fit model
    model: Model = Model(inputs=inception_V3.input, outputs=pred)
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        gen_train,
        validation_data=gen_test,
        validation_steps=1000,
        epochs=epochs,
        verbose=1,
    )
    model.save(model_path)
    return history


def resnet_model(
    generator_img_shape: Tuple[int, int],
    inception_input_shape: Tuple[int, int, int],
    training_dir: str,
    testing_dir: str,
    save_dir: str,
    batch_size: int = 4,
    epochs: int = 5,
    num_classes: int = 2,
) -> Model:
    """
    This function performs transfer learning on the ResNet50V2 model, initially trained on "imagenet" data
    :param generator_img_shape: Length and width of the images
    :param inception_input_shape: Length width and channels of images (channels=1 for greyscale, 3 for colour)
    :param training_dir: directory of the training data
    :param testing_dir: directory of the testing data
    :param save_dir: Where to save the model
    :param batch_size: size of the batch
    :param epochs: number of epochs to train
    :param num_classes: number of classes we are training
    :return: Trained model
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        model_path: str = save_dir
    else:
        model_path = save_dir

    # using image aurgmentation/preprocessing https://keras.io/api/preprocessing/image/
    data_gen_train: ImageDataGenerator = ImageDataGenerator(preprocessing_function=preproc_resnet)
    # generate batches of augmented data
    gen_train: ImageDataGenerator = data_gen_train.flow_from_directory(
        training_dir, target_size=generator_img_shape, batch_size=batch_size, class_mode="categorical"
    )
    gen_test: ImageDataGenerator = data_gen_train.flow_from_directory(
        testing_dir, target_size=generator_img_shape, batch_size=batch_size, class_mode="categorical"
    )
    # assign resnet50V2 model, do not include final layers, we will create those ourselves
    resNet_V2: Model = ResNet50V2(
        weights="imagenet", input_shape=inception_input_shape, include_top=False, pooling=None
    )
    output = resNet_V2.output
    # add pooling layer
    max_pool_output: MaxPool2D = MaxPool2D()(output)
    flatten_output: Flatten = Flatten()(max_pool_output)
    dense_output: Dense = Dense(128, activation="relu")(flatten_output)
    drop_output: Dropout = Dropout(0.1)(dense_output)
    # make prediction using soft max
    pred: Dense = Dense(num_classes, kernel_regularizer=regularizers.l2(0.01), activation="softmax")(drop_output)
    # define, compile, and fit model
    model = Model(inputs=resNet_V2.input, outputs=pred)
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        gen_train,
        validation_data=gen_test,
        validation_steps=1000,
        epochs=epochs,
        verbose=1,
    )
    model.save(model_path)
    return history


def mobilenet_model(
    generator_img_shape: Tuple[int, int],
    inception_input_shape: Tuple[int, int, int],
    training_dir: str,
    testing_dir: str,
    save_dir: str,
    batch_size: int = 4,
    epochs: int = 5,
    num_classes: int = 2,
) -> Model:
    """
    This function performs transfer learning on the InceptionV3 model, initially trained on "imagenet" data
    :param generator_img_shape: Length and width of the images
    :param inception_input_shape: Length width and channels of images (channels=1 for greyscale, 3 for colour)
    :param training_dir: directory of the training data
    :param testing_dir: directory of the testing data
    :param save_dir: Where to save the model
    :param batch_size: size of the batch
    :param epochs: number of epochs to train
    :param num_classes: number of classes we are training
    :return: Trained model
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        model_path: str = save_dir
    else:
        model_path = save_dir

    # using image aurgmentation/preprocessing https://keras.io/api/preprocessing/image/
    data_gen_train: ImageDataGenerator = ImageDataGenerator(preprocessing_function=preproc_mobile)
    # generate batches of augmented data
    gen_train: ImageDataGenerator = data_gen_train.flow_from_directory(
        training_dir, target_size=generator_img_shape, batch_size=batch_size, class_mode="categorical"
    )
    gen_test: ImageDataGenerator = data_gen_train.flow_from_directory(
        testing_dir, target_size=generator_img_shape, batch_size=batch_size, class_mode="categorical"
    )
    # assign mobilenetv2 model, do not include final layers, we will create those ourselves
    MobileNet_V2: Model = MobileNetV2(
        weights="imagenet", input_shape=inception_input_shape, include_top=False, pooling=None
    )
    output = MobileNet_V2.output
    # add pooling layer
    max_pool_output: MaxPool2D = MaxPool2D()(output)
    flatten_output: Flatten = Flatten()(max_pool_output)
    dense_output: Dense = Dense(128, activation="relu")(flatten_output)
    drop_output: Dropout = Dropout(0.1)(dense_output)
    # make prediction using soft max
    pred: Dense = Dense(num_classes, kernel_regularizer=regularizers.l2(0.01), activation="softmax")(drop_output)
    # define, compile, and fit model
    model: Model = Model(inputs=MobileNet_V2.input, outputs=pred)
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])
    history: Model = model.fit(
        gen_train,
        validation_data=gen_test,
        validation_steps=1000,
        epochs=epochs,
        verbose=1,
    )
    model.save(model_path)
    return history


def save_accuracy(history: Model, path: str) -> None:
    """
    Save the accuracy of a given model in a graph
    :param history: trained Model
    :param path: Path to save the plot
    :return: None
    """
    fig: plt.figure = plt.figure(figsize=(10, 10))
    plt.title(path.split("/")[-1])
    plt.plot(history.history["accuracy"])
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train_accuracy"], loc="best")
    fig.patch.set_facecolor("#FFFFFF")  # set background of image to white
    plt.savefig(path)


def save_loss(history: Model, path: str) -> None:
    """
    Save the loss of a given model in a graph
    :param history: trained Model
    :param path: Path to save the plot
    :return: None
    """
    fig: plt.figure = plt.figure(figsize=(10, 10))
    plt.title(path.split("/")[-1])
    plt.plot(history.history["loss"])
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train_loss"], loc="best")
    fig.patch.set_facecolor("#FFFFFF")  # set background of image to white
    plt.savefig(path)


def save_graphs() -> None:
    """
    Saves the accuracy and loss graphs of the 3 models in this script
    """
    save_accuracy(inception_history, str(ROOT) + "/" + "figs/ACC-Inceptionv3.png")
    save_loss(inception_history, str(ROOT) + "/" + "figs/LOSS-Inceptionv3.png")
    save_accuracy(res_history, str(ROOT) + "/" + "figs/ACC-ResNe50tV2.png")
    save_loss(res_history, str(ROOT) + "/" + "figs/LOSS-ResNet50V2.png")
    save_accuracy(mobile_history, str(ROOT) + "/" + "figs/ACC-MobileNetV2.png")
    save_loss(mobile_history, str(ROOT) + "/" + "figs/LOSS-MobileNetV2.png")


def display_class_images(path1: str, path2: str, num_images: int = 3) -> None:
    """
    Given 2 path to diffent classes, this fucntion will provide a 2x3 subplot of
    random class images, with 1 class per row
    """
    fileset1: List[str] = [file for file in glob.glob(path1 + "**/*.jpg", recursive=True)]
    fileset2: List[str] = [file for file in glob.glob(path2 + "**/*.jpg", recursive=True)]
    # referenced https://stackoverflow.com/questions/45993370/matplotlib-indexing-error-on-plotting
    fig, ax = plt.subplots(2, num_images, figsize=(25, 25), squeeze=False)
    for k in range(2):
        for j in range(num_images):
            if k == 0:
                m: int = np.random.randint(low=0, high=len(fileset1))
                img_m: mpimg = mpimg.imread(fileset1[m])
                ax[k, j].imshow(img_m)
                ax[k, j].set_title(str(path1).split("/")[-1])
            else:
                n: int = np.random.randint(low=0, high=len(fileset2))
                img_n: mpimg = mpimg.imread(fileset2[n])
                ax[k, j].imshow(img_n)
                ax[k, j].set_title(str(path2).split("/")[-1])
    fig.patch.set_facecolor("#FFFFFF")
    plt.tight_layout()


def auc_scores(path1: str, path2: str) -> ndarray[float]:
    """

    :param path1:
    :param path2:
    :return:
    """
    # iterate through files in path
    fileset1: ndarray[str] = np.asarray([file for file in glob.glob(path1 + "**/*.jpg", recursive=True)])
    fileset2: ndarray[str] = np.asarray([file for file in glob.glob(path2 + "**/*.jpg", recursive=True)])
    W: int = 165
    H: int = 165
    CHANNELS: int = 3
    N_PIXELS: int = W * H * CHANNELS
    N: int = len(fileset1)
    flats: ndarray = np.zeros((N * 2, N_PIXELS))
    labels: list = []
    # loop over all image files, load them, and save them as numpy arrays
    # also get labels
    fileset_master: ndarray = np.concatenate((fileset1, fileset2))
    for i, f in enumerate(fileset_master):
        img = Image.open(f)
        # flatten array
        label = Path(f).resolve().parent.name  # name of parent folder, i.e. class name like "bibimbap"
        flat: ndarray = np.ravel(img.getdata())
        flats[i, :] = flat
        labels.append(label)
    # convert labels to 0s and 1s
    labels = LabelEncoder().fit_transform(labels)
    scores = []
    for i in range(N_PIXELS):
        flat = flats[:, i]
        roc_auc = roc_auc_score(y_true=labels, y_score=flat)
        scores.append(roc_auc)
    return np.asarray(scores)


def turn_aucs_2D(input_aucs: ndarray[float]) -> ndarray[float]:
    """

    :param input_aucs:
    :return:
    """
    # reshape arrays
    AUCs_3D: ndarray = input_aucs.reshape(165, 165, 3)
    # normalize to values between 0 - 255
    AUCs_3D *= 255.0 / AUCs_3D.max()
    # average across channel numbers to get 2D array
    AUCs_2D: ndarray = np.mean(AUCs_3D, axis=2)
    return AUCs_2D


def save_aucs(to_save: ndarray[float], title: str, save_file: str) -> None:
    """

    :param to_save:
    :param title:
    :param save_file:
    :return:
    """
    figure, axes = plt.subplots()
    figure.set_figheight(15)
    figure.set_figwidth(15)
    axes.set_title(title, fontsize=24)
    figure.patch.set_facecolor("#FFFFFF")
    normalized: StretchOutNormalize = StretchOutNormalize(vmin=0, vmax=255, low=255 / 2, up=255 / 2)
    plt.imshow(to_save, cmap="seismic", norm=normalized)
    plt.clim(0, 255)
    plt.colorbar()
    plt.savefig(save_file)


if __name__ == "__main__":
    # train models
    inception_history: Model = inception_model(
        generator_img_shape=IMG_DIMS,
        inception_input_shape=IMG_SIZE_FULL,
        training_dir=join(ROOT, "data/food-101/train/"),
        testing_dir=join(ROOT, "data/food-101/test/"),
        save_dir=join(ROOT, "outputs"),
        batch_size=6,
        epochs=10,
        num_classes=10,
    )
    res_history: Model = resnet_model(
        generator_img_shape=IMG_DIMS,
        inception_input_shape=IMG_SIZE_FULL,
        training_dir=join(ROOT, "data/food-101/train/"),
        testing_dir=join(ROOT, "data/food-101/test/"),
        save_dir=join(ROOT, "outputs"),
        batch_size=6,
        epochs=10,
        num_classes=10,
    )
    mobile_history: Model = mobilenet_model(
        generator_img_shape=IMG_DIMS,
        inception_input_shape=IMG_SIZE_FULL,
        training_dir=join(ROOT, "data/food-101/train/"),
        testing_dir=join(ROOT, "data/food-101/test/"),
        save_dir=join(ROOT, "outputs"),
        batch_size=6,
        epochs=10,
        num_classes=10,
    )
    # save loss and accuracy graphs
    save_graphs()
    # cuz paper is 2D and variables can't start with numbers
    paper_aucs: ndarray[float] = turn_aucs_2D(
        auc_scores("data/food-101/train/apple_pie", "data/food-101/train/baby_back_ribs")
    )
    # save aucs to a matplotlib colormap
    save_aucs(to_save=paper_aucs, title="AUCs Colourmap - AP and BBR", save_file=str(ROOT) + "/" + "figs/AUC Colourmap")
