# Import necessary libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from aerial_dataset import dataset_creation  # Custom module to create dataset


def augmentation_viewer(dataset, augmentations, batch_size):
    """
    Displays original and augmented images from a dataset.

    Args:
        dataset (tf.data.Dataset): The dataset to visualize.
        augmentations (tf.keras.Model): Augmentation pipeline to apply.
        batch_size (int): Number of images to display from the batch.
    """
    class_names = dataset.class_names  # Get class names from the dataset

    # Take one batch from the dataset
    for images, labels in dataset.take(1):
        for i in range(batch_size):
            # Display the original image
            plt.figure(figsize=(10, 10))
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[np.argmax(labels[i])])
            plt.axis("off")
            plt.show()

            # Display the augmented image
            plt.figure(figsize=(10, 10))
            augmented_image = augmentations(images[i])  # Apply augmentation
            plt.imshow(augmented_image.numpy().astype("uint8"))
            plt.title(class_names[np.argmax(labels[i])])
            plt.axis("off")
            plt.show()


def create_augmentation():
    """
    Creates a sequential augmentation pipeline using Keras preprocessing layers.

    Returns:
        tf.keras.Sequential: The augmentation model.
    """
    all_augmentations = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),  # Randomly flip images
        tf.keras.layers.RandomRotation(0.2)  # Randomly rotate images
    ])

    return all_augmentations


def main():
    """
    Main execution function:
    - Creates dataset
    - Creates augmentations
    - Visualizes original and augmented images
    """
    # Load training and validation datasets with specified image size and batch size
    train_ds, val_ds = dataset_creation(224, 224, 8)

    # Create augmentation pipeline
    augmentations = create_augmentation()

    # Visualize a sample from the dataset with and without augmentation
    augmentation_viewer(train_ds, augmentations, 1)


# Run the script only if this file is executed as the main program
if __name__ == '__main__':
    main()
