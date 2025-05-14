# Deep-Learning-project

# Brain Tumor Detection using 3D U-Net and CNN  

## Overview  
This project implements **3D U-Net** and **CNN-based architectures** for automatic brain tumor detection and segmentation using MRI scans. The goal is to provide an efficient deep learning approach for analyzing **volumetric medical data**, enabling precise tumor identification to assist medical professionals.

## Features  
- **Preprocessing**: Normalization, resizing, and data augmentation for improved model generalization.  
- **3D U-Net**: Used for tumor segmentation within MRI scans.  
- **CNN Classifier**: Determines the presence of tumors using image patches extracted from segmented regions.  
- **Dataset**: Uses publicly available brain tumor MRI datasets.  
- **Evaluation Metrics**: Dice coefficient, accuracy, sensitivity, and specificity for model performance assessment.  

## Installation  
To set up the project, clone this repository and install the required dependencies:  

```bash
git clone https://github.com/your-repo/brain-tumor-detection.git
cd brain-tumor-detection
pip install -r requirements.txt
```

## Usage  
1. **Data Preparation**: Place MRI scans in the `data/` directory and preprocess them using `preprocess.py`.  
2. **Training**: Run the following command to train the 3D U-Net model:  
   ```bash
   python train_unet.py --epochs 50 --batch_size 8
   ```
3. **Tumor Classification**: Use the trained CNN model for tumor detection:  
   ```bash
   python classify_tumor.py --image_path path/to/scan.nii
   ```
4. **Results Visualization**:  
   ```bash
   python visualize.py --image_path path/to/scan.nii
   ```

## Model Architecture  
### **3D U-Net**  
- Utilizes **3D convolutional layers** to process volumetric data.  
- Skip connections help retain spatial information for better segmentation accuracy.  

### **CNN Tumor Classifier**  
- **Pretrained CNN model** (such as ResNet or EfficientNet) is fine-tuned on tumor classification tasks.  
- Outputs the probability of tumor presence.  

## Evaluation  
- **Dice Coefficient**: Measures segmentation accuracy.  
- **Confusion Matrix**: Evaluates tumor classification performance.  
- **ROC Curve**: Assesses model sensitivity and specificity.  

## Results  
The model achieves high segmentation accuracy and tumor detection performance. Example segmented output images can be found in the `results/` directory.

## Contributions  
Feel free to contribute by submitting a **pull request** or reporting issues in the **Issues tab**.  

## License  
This project is open-source and licensed under the **MIT License**.

---
