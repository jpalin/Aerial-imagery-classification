# Import required TensorFlow and Keras modules
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.callbacks import ReduceLROnPlateau  # Optional learning rate scheduler

# Import custom utility functions for dataset creation, plotting, evaluation, and saving
from aerial_dataset import (
    dataset_creation,
    plot_training,
    model_evaluation,
    save_model,
    dataset_creation_custom
)

def create_augmentation():
    """
    Creates a basic image augmentation pipeline using Keras preprocessing layers.

    Returns:
        tf.keras.Sequential: Sequential model containing augmentation layers.
    """
    all_augmentations = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),  # Randomly flip images
        tf.keras.layers.RandomRotation(0.2)  # Randomly rotate images
    ])
    return all_augmentations


def model_creation(augmentation):
    """
    Builds and compiles a convolutional neural network model using sequential API.

    Args:
        augmentation (tf.keras.Sequential): Augmentation pipeline to prepend to the model.

    Returns:
        tf.keras.Model: Compiled CNN model ready for training.
    """
    model = tf.keras.Sequential()

    # Data preprocessing
    model.add(augmentation)  # Include data augmentation as first layer
    model.add(tf.keras.layers.Rescaling(1. / 255))  # Normalize pixel values

    # Convolutional and pooling layers
    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 1)))
    model.add(tf.keras.layers.AveragePooling2D())

    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D())

    model.add(tf.keras.layers.Conv2D(filters=26, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D())

    model.add(tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D())

    model.add(tf.keras.layers.Conv2D(filters=46, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D())

    model.add(tf.keras.layers.Flatten())  # Flatten feature maps into 1D feature vector

    # Fully connected layers
    model.add(tf.keras.layers.Dense(units=120, activation='relu'))
    model.add(tf.keras.layers.Dense(units=84, activation='relu'))

    # Output layer (3 classes, softmax activation for multi-class classification)
    model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def model_training(train_ds, val_ds, model):
    """
    Trains the given model using provided training and validation datasets.

    Args:
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        model (tf.keras.Model): The compiled model to train.

    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    batch_size = 32

    # Early stopping to prevent overfitting
    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=30, mode='max', verbose=0)

    # Save the model weights that give the best validation accuracy
    mcp_save = ModelCheckpoint('.mdl_wts_baseline.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')

    # Optional: Learning rate reduction on plateau (currently commented out)
    # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        batch_size=batch_size,
        epochs=150,
        callbacks=[earlyStopping, mcp_save]
    )

    return history


def main():
    """
    Main function to orchestrate the training pipeline:
    - Load datasets
    - Create model and augmentation
    - Train model
    - Plot training results
    - Evaluate and save the final model
    """
    # Load training and validation datasets
    train_ds, val_ds = dataset_creation(224, 224, 8)

    # Create data augmentation pipeline
    augmentations = create_augmentation()

    # Build and compile the model
    model = model_creation(augmentations)

    # Train the model
    train_history = model_training(train_ds, val_ds, model)

    # Plot accuracy and loss curves
    plot_training(train_history)

    # Evaluate the model on validation data
    model_evaluation(val_ds, model, ".mdl_wts_baseline.hdf5")

    # Save model architecture and weights to disk
    save_model(model, "model_baseline")


# Entry point for script execution
if __name__ == '__main__':
    main()
