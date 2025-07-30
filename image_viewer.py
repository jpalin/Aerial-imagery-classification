import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from aerial_dataset import dataset_creation


def augmentation_viewer(dataset, augmentations, batch_size):
  class_names = dataset.class_names
  for images, labels in dataset.take(1):
    for i in range(batch_size):
      # plot original
      plt.figure(figsize=(10, 10))
      # ax = plt.subplot(6, 6, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[np.argmax(labels[i])])
      plt.axis("off")
      plt.show()
      # plot augmented
      plt.figure(figsize=(10, 10))
      augmented_image = augmentations(images[i])
      plt.imshow(augmented_image.numpy().astype("uint8"))
      plt.title(class_names[np.argmax(labels[i])])
      plt.axis("off")
      plt.show()


def create_augmentation():
  all_augmentations = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2)
  ])

  return all_augmentations

def main():
  train_ds, val_ds = dataset_creation(224,224,8)
  augmentations = create_augmentation()
  augmentation_viewer(train_ds, augmentations, 1)

if __name__ == '__main__':
  main()
