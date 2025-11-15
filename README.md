# Scratch-Detection-on-Text-Images
## Overview
<p align="justify"> 
Scratch-Detection-on-Text-Images is a deep-learning–based project designed to classify whether a text image is clean (**GOOD**) or contains a scratch (**BAD**). 
The project uses an EfficientNetV2-S neural network trained on images of printed text, with additional support for **weakly supervised scratch localization** using Grad-CAM. 
<br><br> This repository includes the complete pipeline: dataset loading, augmentation, model training, evaluation (precision, recall, confusion matrix), inference on single images, and optional scratch localization. 
The goal is to build a robust, production-ready system that reliably detects scratches across multiple text types. </p>

## Abstract
<p align="justify"> This project aims to automatically detect scratches in text images using modern computer vision and deep learning techniques.
The classifier uses EfficientNetV2-S pretrained on ImageNet and fine-tuned on the custom GOOD/BAD dataset.
<br><br> To enhance explainability, Grad-CAM heatmaps are generated to visualize the exact region where scratches are detected, without requiring pixelwise annotations. 
The final model demonstrates high recall and precision on the held-out test set. This system can be integrated into industrial QC pipelines, printing inspections, and automated product verification workflows. </p>

# Table of Contents
Demo Photos
Libraries
Components
Hardware
Code Base
Technologies Used
Result
Conclusion

# Demo Photos
<img width="940" height="453" alt="image" src="https://github.com/user-attachments/assets/c3754cbf-9eb1-4e86-afc6-f616e2bfac99" />
<img width="940" height="431" alt="image" src="https://github.com/user-attachments/assets/a47b81e6-9f43-4783-b796-f654e9f08014" />
<img width="940" height="431" alt="image" src="https://github.com/user-attachments/assets/e886977f-506e-4f19-b091-13218faccc29" />
<img width="940" height="466" alt="image" src="https://github.com/user-attachments/assets/0e4021b2-58e7-4158-aafc-87900e4fba91" />
<img width="940" height="466" alt="image" src="https://github.com/user-attachments/assets/9959b5cc-614f-433f-bc58-a7d7d22cacd6" />


<p align="center"> <img src="examples/good_prediction.png" width="220" /> <img src="examples/bad_prediction.png" width="220" /> <img src="examples/gradcam_bad1.png" width="220" /> <img src="examples/gradcam_bad2.png" width="220" /> </p>

## Libraries
# Library	Description
TensorFlow / Keras:	Model training and EfficientNetV2 backbone
Albumentations:	Advanced image augmentation library
OpenCV:	Image preprocessing & Grad-CAM visualization
NumPy:	Array manipulation & data operations
Scikit-learn:	Evaluation metrics (Precision, Recall, F1, AUC)
Matplotlib:	Plot visualizations for evaluation & Grad-CAM

## Result
<p align="justify"> The model achieved strong quantitative performance on the test set. With robust augmentation and an EfficientNetV2 backbone, the classifier demonstrates high generalization across all three text types in the dataset. </p>

Model Performance 
<img width="624" height="520" alt="image" src="https://github.com/user-attachments/assets/7d6e8bac-3b2b-4613-873d-1053607abd38" />
Confusion Matrix

<img width="803" height="705" alt="image" src="https://github.com/user-attachments/assets/dd393ea4-83e9-4b81-a180-b7edaf32df33" />
ROC Curve

Precision (BAD): 100%

Recall (BAD): 100%

## Conclusion
<p align="justify"> Scratch-Detection-on-Text-Images effectively identifies both clear and scratched text surfaces using a modern convolutional neural network. 
The project leverages EfficientNetV2 for high accuracy and Grad-CAM for explainability, fulfilling both detection and localization requirements. 
<br><br> Future improvements include segmentation-based models like U-Net, synthetic scratch generation, and Mask R-CNN localization. 
With these enhancements, the system can scale into industrial QA workflows for printed surfaces, labels, packaging, and product manufacturing. </p>
