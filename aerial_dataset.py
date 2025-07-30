import tensorflow as tf
import matplotlib.pyplot as plt

def dataset_creation(img_height, img_width, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "C:/aerial_images",
        validation_split=0.2,
        subset="training",
        label_mode="categorical",
        seed=123,
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
    if split == 1.0:
        dataset = tf.keras.utils.image_dataset_from_directory(
            path,
            label_mode="categorical",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        return dataset
    else:
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

  fig1 = plt.gcf()
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.axis(ymin=0.0, ymax=1)
  plt.grid()
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epochs')
  plt.legend(['train', 'validation'])
  plt.show()

  fig2 = plt.gcf()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.axis(ymin=0.0, ymax=2)
  plt.grid()
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.legend(['train', 'validation'])
  plt.show()


def model_evaluation(val_ds, model, weights_file):
  # evaluate on test set
  # Loads the weights
  model.load_weights(weights_file)

  # Re-evaluate the model
  loss, acc = model.evaluate(val_ds, verbose=2)
  print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


def save_model(model, file_name):
  # serialize model to JSON
  model_json = model.to_json()
  with open(file_name+".json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights(file_name+".h5")
  print("Saved model to disk")
