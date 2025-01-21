# Retina Vessel Segmentation using the DRIVE Dataset

## Overview
This project focuses on the segmentation of retinal vessels from fundus images using the DRIVE dataset. The segmentation task is crucial for various medical applications, including the diagnosis of diabetic retinopathy and other retinal diseases. The project implements three different Unet-based architectures, robust preprocessing techniques, and multiple evaluation metrics for performance assessment.

---

## Key Features
- **Dataset:** DRIVE dataset (Digital Retinal Images for Vessel Extraction).
- **Preprocessing Techniques:**
  - Normalization
  - Patching
  - Picture target selection
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

### Prediction
Generate segmentations for new images using:
```bash
python predict.py --model [model_name] --input [path_to_input_image] --output [path_to_output_image]
```

---

## Directory Structure
```
retina-vessel-segmentation/
├── data/                # Dataset folder
├── models/              # Model architectures
├── preprocess/          # Preprocessing scripts
├── postprocess/         # Post-processing scripts
├── train.py             # Training script
├── evaluate.py          # Evaluation script
├── predict.py           # Prediction script
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
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

## Contributing
Contributions are welcome! If you'd like to improve the code or add new features:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add new feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a Pull Request.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- The DRIVE dataset contributors for providing the dataset.
- TensorFlow and OpenCV communities for their amazing tools.

---

## Contact
For any questions or suggestions, feel free to contact:
- **Name:** [Your Name]
- **Email:** [your.email@example.com]
- **GitHub:** [yourusername](https://github.com/yourusername)

