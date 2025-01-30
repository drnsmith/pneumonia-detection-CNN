## AI-Driven Pneumonia Detection Using CNNs and VGG16

### **Project Overview**
This project leverages **Convolutional Neural Networks (CNNs)** to classify chest X-ray images as either **Normal** or **Pneumonia**. It encompasses data pre-processing, model development (including a custom CNN and transfer learning with VGG16), training, evaluation, and performance analysis.

---

### **Motivation**
Pneumonia is a significant health concern worldwide, especially in children and the elderly. Early and accurate detection through chest X-rays is vital for effective treatment. This project aims to:
- Automate pneumonia detection using deep learning techniques.
- Compare the efficacy of a custom-designed CNN with a pre-trained VGG16 model.
- Provide insights into the application of CNNs in medical image classification.

---

### **Dataset Description**
- **Source**: The dataset is sourced from the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset on Kaggle.
- **Structure**:
  - **Training Set**: 5,216 images (3,875 Pneumonia, 1,341 Normal)
  - **Validation Set**: 16 images (all Normal)
  - **Test Set**: 624 images (390 Pneumonia, 234 Normal)
- **Pre-processing**:
  - Images were resized to 224x224 pixels.
  - Applied normalisation to scale pixel values.
  - Data augmentation techniques (e.g., rotation, flipping) were employed to enhance model generalisation.

---

### **Model Architectures**
## **1. Custom CNN**
- **Architecture**:
  - Three convolutional layers with ReLU activation and max-pooling.
  - Two fully connected layers leading to a softmax output.
- **Regularisation**:
  - Dropout layers to prevent overfitting.
- **Optimizer**:
  - Adam optimizer with a learning rate of 0.001.

## **2. VGG16 Transfer Learning**
- **Modification**:
  - Utilized the VGG16 model pre-trained on ImageNet.
  - Replaced the top classifier layers with a custom fully connected network suitable for binary classification.
- **Training Strategy**:
  - Fine-tuned the top layers while keeping the convolutional base frozen initially, then unfroze some layers for further training.

---

## **Training Procedure**
- **Hyperparameters**:
  - Batch size: 32
  - Epochs: 25
- **Loss Function**:
  - Binary Cross-Entropy Loss
- **Data Augmentation**:
  - Applied random rotations, shifts, and flips to augment the training data.
- **Validation**:
  - Monitored validation loss to implement early stopping and prevent overfitting.

---

## **Evaluation Results**
- **Custom CNN**:
  - Accuracy: 85%
  - Recall (Sensitivity): 88%
  - Precision: 83%
  - F1-Score: 85%
- **VGG16 Transfer Learning**:
  - Accuracy: 90%
  - Recall (Sensitivity): 92%
  - Precision: 89%
  - F1-Score: 90%

*Note: Replace the above metrics with actual results from your experiments.*

---

## **Confusion Matrix**
|                  | Predicted Normal | Predicted Pneumonia |
|------------------|------------------|---------------------|
| Actual Normal    | 200              | 34                  |
| Actual Pneumonia | 31               | 359                 |

*Note: Replace the above values with actual results from your experiments.*

---

## **Installation**
### **1. Clone the Repository**
```bash
git clone https://github.com/drnsmith/pneumonia-detection-CNN.git
cd pneumonia-detection-CNN
```
## **2. Install Dependencies**
To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```
---

## Usage
### **1. Prepare the Dataset**
Download the dataset from Kaggle.
Place it in the `data/` directory, organised into the following subdirectories:
```bash
data/
├── train/
├── val/
└── test/
```
### **2. Run Pre-processing**
Execute the `preprocess.py` script to resize images and apply data augmentation:
```bash
python preprocess.py
```

### **3. Train the Model**
Use the `train.py` script to train either the custom CNN or the VGG16 model:
```bash
python train.py
```

### **4. Evaluate the Model**
Run the `evaluate.py` script to assess model performance on the test set:
```bash
python evaluate.py
```
---

## Future Enhancements
### **1. Hyperparameter Tuning**
Experiment with different learning rates, batch sizes, and optimisers to improve performance.

### **2. Additional Data**
Incorporate more diverse datasets to enhance model robustness and generalisation.

### **3. Model Interpretability**
Implement techniques like Grad-CAM or similar methods to visualise regions of interest in X-ray images and improve model explainability.

---
### Productisation  
This pneumonia detection system can be transformed into a **cloud-based AI diagnostic tool** for healthcare providers, offering **real-time analysis of chest X-rays** to support radiologists in early pneumonia detection. By integrating with **hospital information systems (HIS) or telemedicine platforms**, the model can provide **automated pre-screening**, reducing workload and improving diagnostic speed. Additional features like **explainability with Grad-CAM visualisations** can enhance trust and usability in clinical settings.

### Monetisation  
The system can be monetised through a **subscription-based SaaS model**, allowing clinics and hospitals to access **AI-powered X-ray diagnostics on demand**. Licensing the **model as an API** for integration into **existing radiology software** offers another revenue stream. Partnerships with **telehealth companies** and **medical research institutions** can further expand market reach, with premium offerings like **custom model tuning for specific datasets or regulatory compliance adaptations**.

---

## Contributing
We welcome contributions to enhance the functionality and performance of this system. Please fork the repository and submit a pull request with your proposed changes.

---

## Repository History Cleaned

As part of preparing this repository for collaboration, its commit history has been cleaned. This action ensures a more streamlined project for contributors and removes outdated or redundant information in the history. 

The current state reflects the latest progress as of 24/01/2025.

For questions regarding prior work or additional details, please contact the author.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
