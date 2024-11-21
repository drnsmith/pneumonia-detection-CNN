# Pneumonia Detection Using CNNs

This project demonstrates the use of convolutional neural networks (CNNs) to classify chest X-rays as either **Normal** or **Pneumonia**. The project includes:
- Dataset pre-processing and augmentation.
- Custom CNN architecture and pre-trained VGG16 implementation.
- Model training, evaluation, and performance analysis.

## Features
1. **Data Pre-processing**: Handling class imbalance, data augmentation, and splitting datasets into training, validation, and test sets.
2. **Custom CNN Model**: A manually designed CNN trained on chest X-ray data.
3. **Transfer Learning**: Implementation of VGG16 for pneumonia detection.
4. **Evaluation**: Metrics include precision, recall, F1-score, and confusion matrix analysis.

## Project Structure
```kotlin
pneumonia-detection-CNN/
├── README.md
├── requirements.txt
├── data/
│   ├── train/
│   ├── val/
│   ├── test/
├── notebooks/
│   ├── Data_Preprocessing.ipynb
│   ├── Manual_CNN.ipynb
│   ├── VGG16_CNN.ipynb
├── scripts/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
├── results/
│   ├── confusion_matrix.png
│   ├── accuracy_plot.png
│   ├── recall_plot.png
├── LICENSE
└── .gitignore
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pneumonia-detection-CNN.git
   cd pneumonia-detection-CNN
   ```
2.  ```bash
    pip install -r requirements.txt
    ```

## Usage
 - Place your chest X-ray data in the `data/ folder`, organized into subdirectories (`train/, val/, test/`).
 - Run the Jupyter notebooks in the `notebooks/` folder to pre-process data, train models, and evaluate results.

## Results
 - Custom CNN achieved X% accuracy and Y% recall on the test set.
 - VGG16 achieved X% accuracy and Y% recall on the test set.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
