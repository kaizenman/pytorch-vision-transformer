from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader

def dataloaders(train_dir, test_dir, transform, batch_size, num_workers=0):
  train_dataset = ImageFolder(train_dir, transform)
  test_dataset = ImageFolder(test_dir, transform)

  classes = train_dataset.classes
  train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=num_workers
  )
  test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=num_workers
  )

  return train_dataloader, test_dataloader, classes