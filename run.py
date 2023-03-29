import torch
import torchinfo
import data, dataloader, model, measure

from train import train
from torchvision.transforms import transforms
from torchvision import models
from torchmetrics import Accuracy

from torch.optim.lr_scheduler import LinearLR

from pathlib import Path

torch.manual_seed(1)
torch.cuda.manual_seed(1)

# download data
data.download_data(
  'https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip',
  'pizza_steak_sushi.zip',
  'data/pizza_steak_sushi'
)

# transform
WIDTH=224
HEIGHT=224
BATCH_SIZE=32
EPOCHS=7
BASE_LR=3*10**-3
LR_WARMUP=79

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#batched_img, batched_label = next(iter(train_dataloader))
#img, label = batched_img[0], batched_label[0]


# visition_transformer = model.VisionTransformer(batches=BATCH_SIZE, out_features=len(class_names)).to(device)

weights = models.ViT_B_32_Weights.DEFAULT
pretrained_vit = models.vit_b_32(weights).to(device)
transform = weights.transforms()

for parameter in pretrained_vit.parameters():
  parameter.requires_grad=False

pretrained_vit.heads = torch.nn.Linear(in_features=768, out_features=3).to(device)

# torchinfo.summary(
#   model=visition_transformer,
#   input_size=[16, 3, 224, 224],
#   col_names=["input_size", "output_size", "num_params", "trainable"],
#   col_width=20,
#   row_settings=["var_names"]
# )

optimizer = torch.optim.Adam(
  params=pretrained_vit.parameters(),
  lr=BASE_LR
)

# dataloaders
train_dataloader, test_dataloader, class_names = dataloader.dataloaders(
  train_dir='data/pizza_steak_sushi/train',
  test_dir='data/pizza_steak_sushi/test',
  transform=transform,
  batch_size=BATCH_SIZE
)

print(f'Training using device {device}')
losses = train(
  epochs=EPOCHS,
  model=pretrained_vit,
  train_dataloader=train_dataloader,
  test_dataloader=test_dataloader,
  loss_fn=torch.nn.CrossEntropyLoss().to(device),
  accuracy_fn=Accuracy(task='multiclass', num_classes=len(class_names)).to(device),
  optimizer=optimizer,
  device=device
)

model_name='ViT_model'
measure.visualize_learning(losses, model_name=model_name)

model_dir = Path('model')
if not model_dir.is_dir():
  print('{model_dir} does not exist. Creating...')
  model_dir.mkdir(parents=True, exist_ok=True)
  
model_save_path=Path(f'model/{model_name}.pth')
print(f'Saving to model/torch_vision_model.pth...')
torch.save(obj=pretrained_vit.state_dict(), f=model_save_path)
print(f'Saved.')

