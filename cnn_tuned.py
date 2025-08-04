# Import TensorFlow and Keras modules
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import Keras Tuner for hyperparameter optimization
import keras_tuner as kt

# Import custom utility functions for dataset handling, plotting, evaluation, and saving
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
        tf.keras.Sequential: Sequential model with augmentation layers.
    """
    all_augmentations = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2)
    ])
    return all_augmentations


def build_model(hp):
    """
    Builds a CNN model with hyperparameter tuning for one dense layer.

    Args:
        hp (HyperParameters): Hyperparameter search space object provided by Keras Tuner.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    model = tf.keras.Sequential()

    # Add data augmentation
    model.add(create_augmentation())

    # Normalize image pixels to [0, 1]
    model.add(tf.keras.layers.Rescaling(1. / 255))

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

    # Flatten feature maps to feed into dense layers
    model.add(tf.keras.layers.Flatten())

    # Dense (fully connected) layers
    model.add(tf.keras.layers.Dense(units=120, activation='relu'))
    model.add(tf.keras.layers.Dense(units=84, activation='relu'))

    # Tune the number of units in the third dense layer
    units_3 = hp.Int('units_hp', min_value=28, max_value=84, step=8)
    model.add(tf.keras.layers.Dense(units=units_3, activation='relu'))

    # Output layer for 3-class classification
    model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def tuning(train_ds, val_ds, proj_name):
    """
    Uses Keras Tuner's Hyperband algorithm to find optimal number of units in a dense layer.

    Args:
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        proj_name (str): Project name for saving tuning results.

    Returns:
        Tuple: Best hyperparameters and tuner instance.
    """
    batch_size = 64  # Batch size during tuning

    # Initialize Hyperband tuner
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=50,
        factor=8,
        hyperband_iterations=3,
        directory='tuning',
        project_name=proj_name
    )

    # Early stopping callback to prevent overfitting
    stop_early = EarlyStopping(monitor='val_accuracy', patience=15)

    # Start the hyperparameter search
    tuner.search(train_ds, validation_data=val_ds, batch_size=batch_size, epochs=150, callbacks=[stop_early])

    # Retrieve the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The optimal number of units in the new densely-connected
    layer is {best_hps.get('units_hp')}
    """)

    return best_hps, tuner


def train_optimal_model(best_hps, tuner, train_ds, val_ds, batch_size):
    """
    Trains the best model identified by the tuner, then retrains it on the optimal number of epochs.

    Args:
        best_hps (kt.HyperParameters): Best hyperparameters found.
        tuner (kt.Tuner): Tuner instance used for building the model.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        batch_size (int): Batch size for training.

    Returns:
        Tuple: Training history and trained Keras model.
    """
    # Build and train the model with best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_ds, validation_data=val_ds, batch_size=batch_size, epochs=150)

    # Determine the epoch with the best validation accuracy
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % best_epoch)

    # Rebuild and retrain the model using best epoch
    hypermodel = tuner.hypermodel.build(best_hps)
    mcp_save = ModelCheckpoint('.mdl_wts_model_tuned_one.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')

    history = hypermodel.fit(train_ds, validation_data=val_ds, batch_size=batch_size, epochs=best_epoch,
                             callbacks=[mcp_save])

    return history, hypermodel


def main():
    """
    Main pipeline for:
    - Dataset loading
    - Hyperparameter tuning
    - Final training
    - Evaluation and saving
    """
    # Load dataset with default image size and batch size
    train_ds, val_ds = dataset_creation(224, 224, 8)

    # Run hyperparameter tuning
    best_hps, tuner = tuning(train_ds, val_ds, "neuron tuning")

    # Train and evaluate the best model
    train_history, hypermodel = train_optimal_model(best_hps, tuner, train_ds, val_ds, 64)

    # Plot training history
    plot_training(train_history)

    # Evaluate performance on validation set
    model_evaluation(val_ds, hypermodel, '.mdl_wts_model_tuned_one.hdf5')

    # Save final model architecture and weights
    save_model(hypermodel, "model_tuned_one")


# Run the script when executed directly
if __name__ == '__main__':
    main()
