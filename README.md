# Retina Vessel Segmentation using the DRIVE Dataset

## Overview
This project focuses on the segmentation of retinal vessels from fundus images using the DRIVE dataset. The segmentation task is crucial for various medical applications, including the diagnosis of diabetic retinopathy and other retinal diseases. The project implements three different Unet-based architectures, robust preprocessing techniques, and multiple evaluation metrics for performance assessment.

---

## Key Features
- **Dataset:** DRIVE dataset (Digital Retinal Images for Vessel Extraction).
- **Preprocessing Techniques:**
  - Normalization
  - Patching
  - Patch target selection
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Implemented Architectures:**
  1. **Basic Unet:** A classic Unet architecture for biomedical segmentation.
  2. **Attention Unet:** Incorporates attention mechanisms to enhance focus on relevant image features.
  3. **Unet with DenseNet121 Backbone:** Combines Unet with DenseNet121 for enhanced feature extraction and better segmentation results.
- **Post-processing:**
  - Stepping
  - Re-patching to reconstruct segmented images.
- **Evaluation Metrics:**
  - F1 Score
  - Precision
  - Recall
  - Accuracy
  - SSIM (Structural Similarity Index)
  - Additional metrics as needed
- **Frameworks Used:** TensorFlow and OpenCV (cv2).

---

## Getting Started

### Prerequisites
- Python 3.7 or higher
- TensorFlow 2.x
- OpenCV (cv2)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/retina-vessel-segmentation.git
   ```
2. Navigate to the project directory:
   ```bash
   cd retina-vessel-segmentation
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Setup
Download the DRIVE dataset from [here](https://drive.grand-challenge.org/) and place it in the designated `data/` folder within the project directory.

---

## Usage

### Training a Model
To train any of the Unet architectures, use the following command:
```bash
python train.py --model [model_name] --epochs [num_epochs] --batch_size [batch_size]
```
- Replace `[model_name]` with `basic_unet`, `attention_unet`, or `unet_densenet121`.
- Specify the number of epochs and batch size as needed.

### Evaluation
Evaluate the model on the test set using:
```bash
python evaluate.py --model [model_name] --weights [path_to_weights]
```

---

## Results
The models have been evaluated using the DRIVE dataset, and the performance metrics are as follows:
| Model                  | F1 Score | Precision | Recall | Accuracy | SSIM  |
|------------------------|----------|-----------|--------|----------|-------|
| Basic Unet             | XX.XX    | XX.XX     | XX.XX  | XX.XX    | XX.XX |
| Attention Unet         | XX.XX    | XX.XX     | XX.XX  | XX.XX    | XX.XX |
| Unet with DenseNet121  | XX.XX    | XX.XX     | XX.XX  | XX.XX    | XX.XX |

---
