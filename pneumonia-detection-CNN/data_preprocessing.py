import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Resampling function for class imbalance
def resample_images(class_dir, target_count):
    """
    Resamples images to handle class imbalance by oversampling.
    """
    images = os.listdir(class_dir)
    if len(images) < target_count:
        while len(images) < target_count:
            img_to_duplicate = np.random.choice(images)
            src_path = os.path.join(class_dir, img_to_duplicate)
            dst_path = os.path.join(class_dir, f"copy_{len(images)}_{img_to_duplicate}")
            shutil.copy(src_path, dst_path)
    print(f"Class resampled to {target_count} images in {class_dir}")

# Splitting dataset into train, val, and test
def split_class_images(class_dir, val_size=0.2, test_size=0.2):
    """
    Splits images in a class directory into train, val, and test sets.
    """
    images = os.listdir(class_dir)
    train_images, temp_images = train_test_split(images, test_size=(val_size + test_size))
    val_images, test_images = train_test_split(temp_images, test_size=test_size / (val_size + test_size))

    os.makedirs(f"train/{os.path.basename(class_dir)}", exist_ok=True)
    os.makedirs(f"val/{os.path.basename(class_dir)}", exist_ok=True)
    os.makedirs(f"test/{os.path.basename(class_dir)}", exist_ok=True)

    for img in train_images:
        shutil.move(os.path.join(class_dir, img), f"train/{os.path.basename(class_dir)}/")
    for img in val_images:
        shutil.move(os.path.join(class_dir, img), f"val/{os.path.basename(class_dir)}/")
    for img in test_images:
        shutil.move(os.path.join(class_dir, img), f"test/{os.path.basename(class_dir)}/")

    print(f"Split completed for {class_dir}")

# Data augmentation
def augment_data(target_dir, save_dir, augment_count=5000):
    """
    Augments data with ImageDataGenerator and saves augmented images.
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    images = os.listdir(target_dir)
    os.makedirs(save_dir, exist_ok=True)

    for img in images:
        img_path = os.path.join(target_dir, img)
        img_array = np.expand_dims(plt.imread(img_path), axis=0)
        i = 0
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=save_dir, save_prefix="aug", save_format="jpeg"):
            i += 1
            if i >= augment_count // len(images):
                break
    print(f"Data augmentation completed for {target_dir}")


if __name__ == "__main__":
    # Example usage
    resample_images("data/Normal", 5000)
    split_class_images("data/Pneumonia", val_size=0.2, test_size=0.2)
    augment_data("train/Pneumonia", "train/Pneumonia_Augmented", augment_count=5000)
