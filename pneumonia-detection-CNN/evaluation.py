import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

def evaluate_model(model, test_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    test_generator = datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=1, class_mode="binary", shuffle=False)

    # Predictions and ground truth
    predictions = (model.predict(test_generator) > 0.5).astype("int32")
    y_true = test_generator.classes

    # Classification report
    print(classification_report(y_true, predictions, target_names=test_generator.class_indices.keys()))

    # Confusion matrix
    cm = confusion_matrix(y_true, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    from model_training import build_manual_cnn  # Import model architecture
    model = build_manual_cnn(input_shape=(150, 150, 3))
    model.load_weights("path/to/saved/model.h5")  # Load trained weights
    evaluate_model(model, "test")
