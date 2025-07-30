import numpy as np
from PIL import Image
from PIL import ImageEnhance
import pillow_avif
import os


def augment_image(directory, img_class):
  # directory of images to augment
  directory = os.fsencode(directory)

  # loop through all images in directory
  for file in os.listdir(directory):
    # path to image
    path = os.path.join(directory, file)
    print(path)
    img = Image.open(path)
    # apply augmentation
    PIL_image = perform_augmenatation(img, file)
    # create target directory if it does not exist
    target_directory = img_class
    image_path = "C:/test"
    path = os.path.join(image_path, target_directory)

    if not os.path.exists(path):
      os.makedirs(path)
    # save augmented image to new directory
    PIL_image.save(f"{path}/"+str(file)+".jpg")

def perform_augmenatation(img, file):
  if "png" in str(file):
    img = img.convert('RGB')
  # apply augmentation
  enhancer = ImageEnhance.Contrast(img)
  img = enhancer.enhance(2.0)
  # enhancer2 = ImageEnhance.Sharpness(img)
  # img = enhancer2.enhance(3.0)
  # showing resultant image
  # img.show()
  img = np.array(img)
  # Flipping images with Numpy
  flipped_img = np.fliplr(img)
  PIL_image = Image.fromarray(np.uint8(flipped_img)).convert('RGB')
  return PIL_image

def main():
  augment_image("C:/aerial_images_to_be_augmented_final/ocean", "ocean")
  augment_image("C:/aerial_images_to_be_augmented_final/ships", "ships")
  augment_image("C:/aerial_images_to_be_augmented_final/unknown_floating_objects", "unknown_floating_objects")

if __name__ == '__main__':
  main()

