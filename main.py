# import necessory Libraries
from transformers import BeitFeatureExtractor, BeitForSemanticSegmentation
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
import numpy as np
import random
import torch
import os
import sys


class CustomBEITModel:
    def __init__(self, model_name, feature_extractor_name):
        self.feature_extractor = BeitFeatureExtractor.from_pretrained(feature_extractor_name)
        self.model = BeitForSemanticSegmentation.from_pretrained(model_name)

    def process_image(self, image_path):
        try:
            image = Image.open(image_path)
        except Exception as e:
            raise ValueError(f"Error opening the image: {str(e)}")

        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        batch_size, num_labels, height, width = logits.shape

        print("Batch Size:", batch_size)
        print("Number of Labels:", num_labels)
        print("Height of Logits:", height)
        print("Width of Logits:", width)

        return logits
    def visualize_segmentation(self,image_path, logits):
      try:
          image = Image.open(image_path)
      except Exception as e:
          raise ValueError(f"Error opening the image: {str(e)}")
      
      logits_rescale = nn.functional.interpolate(logits,
                                                size=image.size[::-1],  # (height, width)
                                                mode='bilinear',
                                                align_corners=False)
      seg = logits_rescale.argmax(dim=1)[0].cpu()
      color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
      palette = np.array(alternative_palette())

      for label, color in enumerate(palette):
          color_seg[seg == label, :] = color


      color_seg = color_seg[..., ::-1]


      img = np.array(image) * 0.5 + color_seg * 0.5
      img = img.astype(np.uint8)
      save_folder='segmentation_results'
      # Make directory and store segemented Image
      os.makedirs(save_folder, exist_ok=True)

      image_filename = os.path.splitext(os.path.basename(image_path))[0]

      save_path = os.path.join(save_folder, f"{image_filename}_segmentation_result.png")
      plt.imsave(save_path, img)

      plt.figure(figsize=(15, 10))
      plt.imshow(img)
      plt.show()
def alternative_palette():
  num_classes = 150  
  palette = [[random.randint(0, 255) for _ in range(3)] for _ in range(num_classes)]
  return palette
def main_fun():
  try:
    model_name =sys.argv[1]                                            #'microsoft/beit-base-finetuned-ade-640-640'            
    feature_extractor_name =sys.argv[2]                                #'microsoft/beit-base-finetuned-ade-640-640'
    image_path =sys.argv[3]                                                  #'image3.jpg'

    custom_beit_model = CustomBEITModel(model_name, feature_extractor_name)
    result_logits = custom_beit_model.process_image(image_path)
    custom_beit_model.visualize_segmentation(image_path, result_logits)


    print("Logits:", result_logits[0])

  except Exception as e:
    print(f"An error occurred: {str(e)}")

if __name__ == "__main__":                       # Main 
  main_fun()

