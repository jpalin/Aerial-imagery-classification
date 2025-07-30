import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from aerial_dataset import dataset_creation, plot_training, model_evaluation, save_model, dataset_creation_custom


def create_augmentation():
  all_augmentations = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2)
  ])

  return all_augmentations

def build_model(hp):  # random search passes this hyperparameter() object
    # define architecture layer by layer
    model = tf.keras.Sequential()
    # add augmentation
    model.add(create_augmentation())
    # apply normalisation
    model.add(tf.keras.layers.Rescaling(1. / 255))
    # Convolutional layers and AveragePooling
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
    # flatten output before entering fully connected layers
    model.add(tf.keras.layers.Flatten())
    # fully connected layers, the third is to be tuned
    model.add(tf.keras.layers.Dense(units=120, activation='relu'))
    model.add(tf.keras.layers.Dense(units=84, activation='relu'))
    units_3 = hp.Int('units_hp', min_value = 28, max_value = 84, step = 8)
    model.add(tf.keras.layers.Dense(units = units_3, activation='relu'))
    # output layer that uses softmax to provide a classification
    model.add(tf.keras.layers.Dense(units=3, activation='softmax'))
    # compile model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def tuning(train_ds, val_ds, proj_name):
    # set batch size for tuning
    batch_size = 64
    # define Hyperband tuner used to tune the fully connected layer
    tuner = kt.Hyperband(build_model,
                         objective='val_accuracy',
                         max_epochs=50,
                         factor=8,
                         hyperband_iterations=3,
                         directory='tuning',
                         project_name=proj_name)
    # early stopping if no improvement in validation accuracy is shown after a set period
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15)
    # start search with Hyperband
    tuner.search(train_ds, validation_data=val_ds, batch_size=batch_size, epochs=150, callbacks=[stop_early])
    # get the optimal hyperparameters identified
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    # print the best number of neurons
    print(f"""
    The optimal number of units in the new densely-connected
    layer is {best_hps.get('units_hp')}
    """)
    return best_hps, tuner


def train_optimal_model(best_hps, tuner, train_ds, val_ds, batch_size):
    # Build the model with the optimal hyperparameters and train it on the data for 150 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_ds, validation_data=val_ds, batch_size=batch_size, epochs=150)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    mcp_save = ModelCheckpoint('.mdl_wts_model_tuned_one.hdf5', save_best_only=True, monitor='val_accuracy',
                               mode='max')

    # Retrain the model
    history = hypermodel.fit(train_ds, validation_data=val_ds, batch_size=batch_size, epochs=best_epoch,
                             callbacks=[mcp_save])
    return history, hypermodel


def main():
  train_ds, val_ds = dataset_creation(224, 224, 8)
  best_hps, tuner = tuning(train_ds, val_ds, "neuron tuning")
  train_history, hypermodel = train_optimal_model(best_hps, tuner, train_ds, val_ds, 64)
  plot_training(train_history)
  model_evaluation(val_ds, hypermodel, '.mdl_wts_model_tuned_one.hdf5')
  save_model(hypermodel, "model_tuned_one")


if __name__ == '__main__':
  main()
