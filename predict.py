import argparse
import torch
import torchvision
import json

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--image', type=str, required=True)

args = parser.parse_args()

model_path = Path(args.model)
image_path = Path(args.image)

# Read the JSON data from the file
with open("classes.json", "r") as file:
    json_data = file.read()
class_names = json.loads(json_data)

weights = torchvision.models.ViT_B_32_Weights.DEFAULT
transform = weights.transforms()
loaded_model = torchvision.models.vit_b_32(weights).to(device)

# freeze the model
for param in loaded_model.parameters():
  param.requires_grad = False

loaded_model.heads = torch.nn.Sequential(
  torch.nn.Linear(in_features=768, out_features=3, bias=True)
).to(device)


loaded_model.eval()
loaded_model.load_state_dict(torch.load(model_path), strict=False)

with torch.inference_mode():
  img = Image.open(image_path)
  transformed_image = transform(img)
  target_image_pred = loaded_model(transformed_image.unsqueeze(dim=0).to(device))
  displayed_image = transformed_image.permute(1, 2, 0).to('cpu')
  f = plt.figure(figsize=(9,9))
  plt.imshow(displayed_image)
  logit = torch.softmax(target_image_pred, dim=1).argmax(dim=1)
  plt.title(label={class_names[str(logit.item())]})
  plt.axis(False)
  f.savefig('predicted.png')
  plt.close(f)

  print(f'Predicted class: {class_names[str(logit.item())]}')