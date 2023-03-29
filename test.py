import torch
import data, dataloader, model
from torchvision.transforms import transforms

from pathlib import Path

# download data
data.download_data(
  'https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip',
  'pizza_steak_sushi.zip',
  'data/pizza_steak_sushi'
)

# transform
WIDTH=224
HEIGHT=224
BATCH_SIZE=16

transform = transforms.Compose([
  transforms.Resize(size=(WIDTH, HEIGHT)),
  transforms.ToTensor()
])

# dataloaders
train_dataloader, test_dataloader, class_names = dataloader.dataloaders(
  train_dir='data/pizza_steak_sushi/train',
  test_dir='data/pizza_steak_sushi/test',
  transform=transform,
  batch_size=BATCH_SIZE
)

batched_img, batched_label = next(iter(train_dataloader))
img, label = batched_img[0], batched_label[0]

num_colors = 3
# Vision transformer model
visition_transformer = model.VisionTransformer(num_colors, len(class_names))
print(f'Done.')

#TODO training...