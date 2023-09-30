import numpy as np
import pandas as pd

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from torchvision import transforms

def convert_to_cuda(images, device):
  transform = transforms.Compose([transforms.ToTensor()])
  return [transform(image).to(device) for image in images]

class CLIP:
  def __init__(self, labels, model_name="openai/clip-vit-large-patch14"):
    "Initializes a zero-shot image model."
    self.labels = labels
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load CLIP
    self.model = CLIPModel.from_pretrained(model_name)
    self.model.to(self.device)
    self.processor = CLIPProcessor.from_pretrained(model_name)
  def process(self, images):
    "Processes a set of images and temporarily stores results."
    images = convert_to_cuda(images, device=self.device)
    inputs = self.processor(text=self.labels, images=images, return_tensors="pt", padding=True)
    inputs = inputs.to(self.device)
    outputs = self.model(**inputs)
    self.logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    del images
  def logits(self):
    return self.logits_per_image
  def probs(self):
    "Returns the probability that each image is in each category."
    probs = self.logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    # Probabilities
    return probs.detach().cpu().numpy()
  def classes(self):
    "Returns the most likely classes each image is in."
    return [self.labels[i] for i in np.argmax(self.probs(), axis=1)] # Most likely class

