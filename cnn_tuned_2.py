# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import custom modules for dataset management and model utilities
from aerial_dataset import (
    dataset_creation,
    plot_training,
    model_evaluation,
    save_model,
    dataset_creation_custom
)


def create_augmentation():
    """
    Creates and returns an image augmentation pipeline using Keras preprocessing layers.
    
    Returns:
        tf.keras.Sequential: A sequential model with augmentation layers.
    """
    all_augmentations = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),  # Randomly flip images
        tf.keras.layers.RandomRotation(0.2)  # Randomly rotate images by up to 20%
    ])
    return all_augmentations


def build_model(hp):
    """
    Builds a CNN model with two hyperparameter-tuned dense layers using Keras Tuner.

    Args:
        hp (kt.HyperParameters): Hyperparameter object from Keras Tuner.

    Returns:
        tf.keras.Model: Compiled CNN model.
    """
    model = tf.keras.Sequential()

    # Add augmentation and normalization
    model.add(create_augmentation())
    model.add(tf.keras.layers.Rescaling(1. / 255))  # Normalize pixel values to [0, 1]

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

    model.add(tf.keras.layers.Flatten())  # Flatten feature maps into 1D vector

    # Dense layers (two of them are hyperparameter-tuned)
    model.add(tf.keras.layers.Dense(units=120, activation='relu'))
    model.add(tf.keras.layers.Dense(units=84, activation='relu'))

    # Tune number of units in 3rd and 4th dense layers
    units_3 = hp.Int('units_hp3', min_value=28, max_value=84, step=8)
    model.add(tf.keras.layers.Dense(units=units_3, activation='relu'))

    units_4 = hp.Int('units_hp4', min_value=28, max_value=84, step=8)
    model.add(tf.keras.layers.Dense(units=units_4, activation='relu'))

    # Output layer for 3-class classification
    model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def tuning(train_ds, val_ds, proj_name):
    """
    Performs hyperparameter tuning on the model using Hyperband strategy.

    Args:
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        proj_name (str): Project name for saving tuner results.

    Returns:
        Tuple[kt.HyperParameters, kt.Tuner]: Best hyperparameters and tuner object.
    """
    batch_size = 64

    # Initialize Keras Tuner using Hyperband algorithm
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=50,
        factor=8,
        hyperband_iterations=3,
        directory='tuning',
        project_name=proj_name
    )

    # Early stopping to prevent overfitting
    stop_early = EarlyStopping(monitor='val_accuracy', patience=15)

    # Start tuning
    tuner.search(train_ds, validation_data=val_ds, batch_size=batch_size, epochs=150, callbacks=[stop_early])

    # Retrieve the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The optimal number of units in the first new densely-connected
    layer is {best_hps.get('units_hp3')} and in the second is {best_hps.get('units_hp4')}
    """)

    return best_hps, tuner


def train_optimal_model(best_hps, tuner, train_ds, val_ds, batch_size):
    """
    Trains the model using the best hyperparameters and determines the best epoch.

    Args:
        best_hps (kt.HyperParameters): Best hyperparameters from tuner.
        tuner (kt.Tuner): Tuner used to build models.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        batch_size (int): Batch size for training.

    Returns:
        Tuple: Training history and trained model.
    """
    # Initial training to find best epoch
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_ds, validation_data=val_ds, batch_size=batch_size, epochs=150)

    # Identify best epoch based on validation accuracy
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % best_epoch)

    # Rebuild model and retrain using only best epoch
    hypermodel = tuner.hypermodel.build(best_hps)
    mcp_save = ModelCheckpoint('.mdl_wts_best_model_tuned_2_augmented.hdf5',
                               save_best_only=True,
                               monitor='val_accuracy',
                               mode='max')

    history = hypermodel.fit(train_ds, validation_data=val_ds, batch_size=batch_size,
                             epochs=best_epoch, callbacks=[mcp_save])

    return history, hypermodel


def main():
    """
    Main function to run the model training pipeline:
    - Dataset loading
    - Hyperparameter tuning
    - Final training
    - Plotting
    - Evaluation
    - Saving the model
    """
    # Load training and validation datasets, including augmented images
    train_ds, val_ds = dataset_creation_custom(
        split=0.8,
        img_height=224,
        img_width=224,
        batch_size=8,
        path="C:/aerial_images_with_added_augmented_set"
    )

    # Perform hyperparameter tuning
    best_hps, tuner = tuning(train_ds, val_ds, "neuron tuning 2 augmented")

    # Train the model with the best hyperparameters
    train_history, hypermodel = train_optimal_model(best_hps, tuner, train_ds, val_ds, 64)

    # Plot training history (accuracy/loss curves)
    plot_training(train_history)

    # Evaluate the best saved model
    model_evaluation(val_ds, hypermodel, '.mdl_wts_best_model_tuned_2_augmented.hdf5')

    # Save the final model architecture and weights
    save_model(hypermodel, "model_tuned_two_augmented_set")


# Entry point of the script
if __name__ == '__main__':
    main()
