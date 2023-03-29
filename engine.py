import torch

def train_step(model, dataloader, loss_fn, accuracy_fn, optimizer, lr_scheduler, device):
  model.train()
  train_loss = 0
  train_accuracy = 0
  for _, (X, y) in enumerate(dataloader):
    X,y = X.to(device), y.to(device)
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_accuracy += accuracy_fn(torch.softmax(y_pred, dim=1).argmax(dim=1), y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if lr_scheduler is not None:
      lr_scheduler.step()

  train_loss /= len(dataloader)
  train_accuracy /= len(dataloader)
  return train_loss, train_accuracy

def test_step(model, dataloader, loss_fn, accuracy_fn, device):
  model.eval()
  with torch.inference_mode():
    test_loss = 0
    test_accuracy = 0
    for batch, (X, y) in enumerate(dataloader):
      X,y = X.to(device), y.to(device)
      y_pred = model(X)
      test_loss += loss_fn(y_pred, y)
      test_accuracy += accuracy_fn(torch.softmax(y_pred, dim=1).argmax(dim=1), y)
    test_loss /= len(dataloader)
    test_accuracy /= len(dataloader)
    return test_loss, test_accuracy