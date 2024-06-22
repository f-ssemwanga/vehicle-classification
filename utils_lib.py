from random import shuffle, choice, randint
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import cv2
import math


def load_rgb_data(
    IMAGE_DIRECTORY,
    IMAGE_SIZE,
    directory_depth=2,
    max_number_of_images=None,
    shuffle_data=True,
):
    """
    Load RGB images from the specified directory and subdirectories.

    Args:
    - IMAGE_DIRECTORY (str): The root directory containing images.
    - IMAGE_SIZE (int): The target size to resize images (square).
    - directory_depth (int): The depth of directory to use as label.
    - max_number_of_images (int): Maximum number of images to load.
    - shuffle_data (bool): Whether to shuffle the data.

    Returns:
    - images (numpy.ndarray): Loaded and processed images.
    - labels (numpy.ndarray): Corresponding labels.
    """
    print("Loading images...", flash=True)
    data = []
    count = 0

    for folder, _, file_names in os.walk(IMAGE_DIRECTORY):
        if len(file_names) != 0:
            label = (
                folder.split(os.sep)[-directory_depth]
                if directory_depth > 0
                else os.path.basename(folder)
            )
            for image_name in file_names:
                image_path = os.path.join(folder, image_name)
                if ".DS_Store" not in image_path:
                    try:
                        img = Image.open(image_path).convert("RGB")
                        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
                        data.append([np.array(img), label])
                        count += 1
                        if count % 50 == 0:
                            print(f"Loaded {count} images so far ...")
                        if max_number_of_images and count >= max_number_of_images:
                            print("Reached maximum number of images. Preparing data...")
                            return _prepare_data(data, IMAGE_SIZE, IMAGE_DIRECTORY)
                    except Exception as e:
                        print(f"Cannot load {image_path}: {e}")

    print(f"Number of images loaded: {count}")
    if shuffle_data:
        random.shuffle(data)
        print("Dataset shuffled.")

    print("Preparing data after full iteration...")
    return _prepare_data(data, IMAGE_SIZE, IMAGE_DIRECTORY)


def _prepare_data(data, IMAGE_SIZE, IMAGE_DIRECTORY):
    """
    Prepare images and labels as numpy arrays.

    Args:
    - data (list): List of image and label pairs.
    - IMAGE_SIZE (int): Size of the image.

    Returns:
    - images (numpy.ndarray): Processed images.
    - labels (numpy.ndarray): Corresponding labels.
    """
    print("Convert images and labels into a numpy array")
    images = np.array([item[0] for item in data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
    labels = np.array([item[1] for item in data])
    print(f"Data Shape: {images.shape}")
    print(f"Labels Shape: {labels.shape}")
    print(f"Unique Labels: {np.unique(labels)}")
    print(f"Loading dataset completed from path: {IMAGE_DIRECTORY}")
    return images, labels


def normalize_data(dataset):
    """
    Normalize dataset to range [0, 1].

    Args:
    - dataset (numpy.ndarray): Dataset to normalize.

    Returns:
    - dataset (numpy.ndarray): Normalized dataset.
    """
    print("Normalizing data...")
    return dataset / 255.0


def display_image(trainX, trainY, index=0):
    """
    Display a single image with its label.

    Args:
    - trainX (numpy.ndarray): Dataset of images.
    - trainY (numpy.ndarray): Corresponding labels.
    - index (int): Index of the image to display.
    """
    plt.imshow(trainX[index])
    print("Label =", np.squeeze(trainY[index]))
    print("Image shape:", trainX[index].shape)


def display_one_image(one_image, its_label):
    """
    Display a single image with its label.

    Args:
    - one_image (numpy.ndarray): Image to display.
    - its_label (str): Label of the image.
    """
    plt.imshow(one_image)
    print("Label =", its_label)
    print("Image shape:", one_image.shape)


def display_dataset_shape(X, Y):
    """
    Display the shapes of the dataset and labels.

    Args:
    - X (numpy.ndarray): Dataset of images.
    - Y (numpy.ndarray): Corresponding labels.
    """
    print("Shape of images:", X.shape)
    print("Shape of labels:", Y.shape)


def plot_sample_from_dataset(images, labels, rows=5, columns=5, width=8, height=8):
    """
    Plot a sample grid of images from the dataset.

    Args:
    - images (numpy.ndarray): Dataset of images.
    - labels (numpy.ndarray): Corresponding labels.
    - rows (int): Number of rows in the grid.
    - columns (int): Number of columns in the grid.
    - width (int): Width of the figure.
    - height (int): Height of the figure.
    """
    plt.figure(figsize=(width, height))
    for i in range(rows * columns):
        plt.subplot(rows, columns, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(labels[i])
    plt.show()


def display_dataset_folders(path):
    """
    Display sorted list of folders in the dataset directory.

    Args:
    - path (str): Path to the dataset directory.
    """
    classes = sorted(os.listdir(path))
    print(classes)


def get_data_distribution(IMAGE_DIRECTORY, output_file=None):
    """
    Get distribution of data and optionally save to CSV.

    Args:
    - IMAGE_DIRECTORY (str): Directory containing the images.
    - output_file (str): File to save the distribution statistics.

    Returns:
    - stats (list): List of statistics for each image.
    """
    print("Loading images...")
    stats = []

    directories = next(os.walk(IMAGE_DIRECTORY))[1]
    for directory_name in directories:
        print(f"Loading {directory_name}")
        images_file_names = next(
            os.walk(os.path.join(IMAGE_DIRECTORY, directory_name))
        )[2]
        print(f"Loading {len(images_file_names)} files from {directory_name} class...")
        for image_name in images_file_names:
            image_path = os.path.join(IMAGE_DIRECTORY, directory_name, image_name)
            try:
                img = Image.open(image_path).convert("RGB")
                width, height = img.size
                size_kb = os.stat(image_path).st_size / 1000
                stats.append(
                    [
                        directory_name,
                        os.path.basename(image_name),
                        width,
                        height,
                        size_kb,
                    ]
                )
            except Exception as e:
                print(f"Cannot load {image_path}: {e}")

    if output_file:
        stats_df = pd.DataFrame(
            stats, columns=["Class", "Filename", "Width", "Height", "Size_in_KB"]
        )
        stats_df.to_csv(output_file, index=False)
        print(f"Stats collected and saved in {output_file}")
    else:
        print("Stats collected")

    return stats


def plot_dataset_distribution(
    stats,
    num_cols=5,
    width=10,
    height=5,
    histogram_bins=10,
    histogram_range=[0, 1000],
    figure_padding=4,
):
    """
    Plot distribution of dataset based on image sizes.

    Args:
    - stats (list): List of statistics for each image.
    - num_cols (int): Number of columns in the grid.
    - width (int): Width of the figure.
    - height (int): Height of the figure.
    - histogram_bins (int): Number of bins in the histogram.
    - histogram_range (list): Range of the histogram.
    - figure_padding (int): Padding between subplots.
    """
    stats_df = pd.DataFrame(
        stats, columns=["Class", "Filename", "Width", "Height", "Size_in_KB"]
    )
    list_sizes = stats_df["Size_in_KB"]
    number_of_classes = stats_df["Class"].nunique()
    print(f"{number_of_classes} classes found in the dataset")

    class_names = ["Whole dataset"]
    list_sizes_per_class = [list_sizes]

    for class_name in stats_df["Class"].unique():
        class_avg_size = list_sizes[stats_df["Class"] == class_name].mean()
        print(f"Sizes of class {class_name} have an average size of {class_avg_size}")
        list_sizes_per_class.append(list_sizes[stats_df["Class"] == class_name])
        class_names.append(class_name)

    class_count_dict = stats_df["Class"].value_counts().to_dict()
    print(class_count_dict)

    num_rows = math.ceil((number_of_classes + 1) / num_cols)
    if number_of_classes < num_cols:
        num_cols = number_of_classes + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(width, height))
    fig.tight_layout(pad=figure_padding)

    for i in range(num_rows):
        for j in range(num_cols):
            idx = num_cols * i + j
            if idx < len(list_sizes_per_class):
                axes[i, j].hist(
                    list_sizes_per_class[idx],
                    bins=histogram_bins,
                    range=histogram_range,
                )
                axes[i, j].set_xlabel("Image size (in KB)", fontweight="bold")
                axes[i, j].set_title(f"{class_names[idx]} images", fontweight="bold")

    plt.figure()
    plt.bar(class_count_dict.keys(), class_count_dict.values())
    for index, class_name in enumerate(class_count_dict):
        plt.text(
            index,
            class_count_dict[class_name] + 1,
            str(class_count_dict[class_name]),
            ha="center",
        )
    plt.show()
