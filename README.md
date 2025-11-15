## Scratch-Detection-on-Text-Images
# Overview
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

Replace the image paths with Grad-CAM and classification result samples from your model.

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
Model Performance (Example)

Precision (BAD): 0.92

Recall (BAD): 0.89



<p align="center"> <img src="examples/confusion_matrix.png" width="350" /> </p> <p align="justify"> Grad-CAM produced accurate heatmaps showing scratch locations even in subtle cases. </p>

## Conclusion
<p align="justify"> Scratch-Detection-on-Text-Images effectively identifies both clear and scratched text surfaces using a modern convolutional neural network. 
The project leverages EfficientNetV2 for high accuracy and Grad-CAM for explainability, fulfilling both detection and localization requirements. 
<br><br> Future improvements include segmentation-based models like U-Net, synthetic scratch generation, and Mask R-CNN localization. 
With these enhancements, the system can scale into industrial QA workflows for printed surfaces, labels, packaging, and product manufacturing. </p>
