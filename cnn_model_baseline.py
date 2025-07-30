import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from aerial_dataset import dataset_creation, plot_training, model_evaluation, save_model, dataset_creation_custom

def create_augmentation():
  all_augmentations = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2)
  ])

  return all_augmentations

def model_creation(augmentation):
  model = tf.keras.Sequential()
  model.add(augmentation)
  model.add(tf.keras.layers.Rescaling(1. / 255))

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

  model.add(tf.keras.layers.Flatten())

  model.add(tf.keras.layers.Dense(units=120, activation='relu'))

  model.add(tf.keras.layers.Dense(units=84, activation='relu'))

  model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

  model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

  return model

def model_training(train_ds, val_ds, model):
  batch_size = 32

  earlyStopping = EarlyStopping(monitor='val_accuracy', patience=30, verbose=0, mode='max')
  mcp_save = ModelCheckpoint('.mdl_wts_baseline.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
  # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

  history = model.fit(train_ds, validation_data=val_ds, batch_size=batch_size, epochs=150,
                      callbacks=[earlyStopping, mcp_save])

  return history

def main():
  train_ds, val_ds = dataset_creation(224, 224, 8)
  augmentations = create_augmentation()
  model = model_creation(augmentations)
  train_history = model_training(train_ds, val_ds, model)
  plot_training(train_history)
  model_evaluation(val_ds, model, ".mdl_wts_baseline.hdf5")
  save_model(model, "model_baseline")


if __name__ == '__main__':
  main()

