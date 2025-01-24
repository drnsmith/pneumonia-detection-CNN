from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Manual CNN
def build_manual_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    return model

# Pre-trained VGG16
def build_vgg16(input_shape):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    base_model.trainable = False  # Freeze VGG16 layers
    return model

# Training function
def train_model(model, train_dir, val_dir, batch_size=32, epochs=10):
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=batch_size, class_mode="binary")
    val_generator = datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=batch_size, class_mode="binary")

    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(train_generator, validation_data=val_generator, epochs=epochs)
    return history

if __name__ == "__main__":
    input_shape = (150, 150, 3)

    # Train manual CNN
    manual_cnn = build_manual_cnn(input_shape)
    train_model(manual_cnn, "train", "val", epochs=10)

    # Train VGG16
    vgg16_model = build_vgg16(input_shape)
    train_model(vgg16_model, "train", "val", epochs=10)

