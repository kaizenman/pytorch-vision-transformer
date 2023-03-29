import torch
import torchinfo
import data, dataloader, model

from train import train
from torchvision.transforms import transforms
from torchmetrics import Accuracy

from torch.optim.lr_scheduler import LinearLR

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
EPOCHS=7
BASE_LR=8**-4
LR_WARMUP=79

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

visition_transformer = model.VisionTransformer(batches=16, out_features=len(class_names)).to(device)

# torchinfo.summary(
#   model=visition_transformer,
#   input_size=[16, 3, 224, 224],
#   col_names=["input_size", "output_size", "num_params", "trainable"],
#   col_width=20,
#   row_settings=["var_names"]
# )

optimizer = torch.optim.Adam(params=visition_transformer.parameters(), lr=BASE_LR)

print(f'Training using device {device}')

results = train(
  epochs=1,
  model=visition_transformer,
  train_dataloader=train_dataloader,
  test_dataloader=test_dataloader,
  loss_fn=torch.nn.CrossEntropyLoss().to(device),
  accuracy_fn=Accuracy(task='multiclass', num_classes=len(class_names)).to(device),
  optimizer=optimizer,
  device=device
)

model_save_path=Path('model/torch_vision_model.pth')
print(f'Saving to model/torch_vision_model.pth...')
torch.save(obj=visition_transformer.state_dict(), f=model_save_path)
print(f'Saved.')