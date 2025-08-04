# Import necessary libraries
import tensorflow as tf
import matplotlib.pyplot as plt

def dataset_creation(img_height, img_width, batch_size):
    """
    Loads and splits the dataset from a fixed directory path into training and validation sets.

    Args:
        img_height (int): Height of the input images.
        img_width (int): Width of the input images.
        batch_size (int): Number of images per batch.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "C:/aerial_images",  # Path to the image directory
        validation_split=0.2,  # 20% for validation
        subset="training",
        label_mode="categorical",  # One-hot encoded labels
        seed=123,  # Random seed for reproducibility
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "C:/aerial_images",
        validation_split=0.2,
        subset="validation",
        label_mode="categorical",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return train_ds, val_ds


def dataset_creation_custom(split, img_height, img_width, batch_size, path):
    """
    Loads a dataset from a given path, with optional train/validation split.

    Args:
        split (float): Proportion for validation split (1.0 = no split).
        img_height (int): Height of input images.
        img_width (int): Width of input images.
        batch_size (int): Batch size.
        path (str): Path to the image directory.

    Returns:
        Union[tf.data.Dataset, Tuple[tf.data.Dataset, tf.data.Dataset]]:
            - A single dataset if split == 1.0
            - A tuple of (train_ds, val_ds) otherwise
    """
    if split == 1.0:
        # Load the entire dataset without splitting
        dataset = tf.keras.utils.image_dataset_from_directory(
            path,
            label_mode="categorical",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        return dataset
    else:
        # Load and split the dataset into training and validation sets
        train_ds = tf.keras.utils.image_dataset_from_directory(
            path,
            validation_split=split,
            subset="training",
            label_mode="categorical",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        val_ds = tf.keras.utils.image_dataset_from_directory(
            path,
            validation_split=split,
            subset="validation",
            label_mode="categorical",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        return train_ds, val_ds


def plot_training(history):
    """
    Plots training and validation accuracy and loss over epochs.

    Args:
        history (tf.keras.callbacks.History): History object returned by model.fit().
    """
    # Plot model accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'])
    plt.show()

    # Plot model loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylim(0.0, 2.0)
    plt.grid(True)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'])
    plt.show()


def model_evaluation(val_ds, model, weights_file):
    """
    Loads trained weights into a model and evaluates on the validation dataset.

    Args:
        val_ds (tf.data.Dataset): Validation dataset.
        model (tf.keras.Model): The Keras model to evaluate.
        weights_file (str): Path to the model weights file (HDF5 format).
    """
    # Load trained weights
    model.load_weights(weights_file)

    # Evaluate model performance on validation data
    loss, acc = model.evaluate(val_ds, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


def save_model(model, file_name):
    """
    Saves the model architecture to JSON and weights to HDF5 format.

    Args:
        model (tf.keras.Model): The model to save.
        file_name (str): Base name for the output files (no extension).
    """
    # Save model architecture to a JSON file
    model_json = model.to_json()
    with open(file_name + ".json", "w") as json_file:
        json_file.write(model_json)

    # Save model weights to HDF5
    model.save_weights(file_name + ".h5")
    print("Saved model to disk")
