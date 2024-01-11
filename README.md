# BEiT Image Segmentation

## Overview

This repository focuses on image segmentation using the BEiT (Vision Transformer) model. BEiT, a Vision Transformer (ViT) with a transformer encoder architecture resembling BERT, undergoes pretraining on a diverse collection of images in a self-supervised manner, specifically on ImageNet-21k at a resolution of 224x224 pixels.
### Model Description
The BEiT model serves as a Vision Transformer (ViT) with a BERT-like transformer encoder architecture. In comparison to the original ViT model, BEiT is pretrained on ImageNet-21k, enhancing its capabilities as a feature extractor for various computer vision tasks.

### Feature Extractor
Initialize the feature extractor using the following code snippet:
```python 
feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-finetuned-ade-640-640')

```
### Process Flow:
In the context of image segmentation an input image is passed through the model.The model extracts features from the image.Labels each pixel in the image, identifying objects and regions assigning specific labels to different regions within the image. The model's ability to capture complex visual patterns and semantic relationships enables it to perform accurate segmentation, making it a powerful tool for various computer vision tasks.
### Dependencies
Ensure you have the following libraries installed:
Torch Library:
```bash
pip install torch torchvision

```
Transformers Library:
```bash
pip install transformers

```
PILL Library:
```bash
pip install Pillow
```
### Running the Script
Execute the following command to run the script:
```python
python3 main.py --file_name your_file_name --model_name your_model_name --feature_extractor your_feature_extractor --image_path path/to/your/image.jpg

```
