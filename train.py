import engine
import torch

def train(
  epochs,
  model,
  train_dataloader,
  test_dataloader,
  loss_fn,
  accuracy_fn,
  optimizer,
  device,
  lr_scheduler=None,
):
  parameters = {'train_loss': [],
                'train_accuracy': [],
                'test_loss': [],
                'test_accuracy': []}

  for epoch in range(epochs):
    train_loss, train_accuracy = engine.train_step(
      model, train_dataloader, loss_fn, accuracy_fn, optimizer, lr_scheduler, device)
    test_loss, test_accuracy = engine.test_step(
      model, test_dataloader, loss_fn, accuracy_fn, device
    )

    parameters['train_loss'].append(train_loss.item())
    parameters['train_accuracy'].append(train_accuracy.item())
    parameters['test_loss'].append(test_loss.item())
    parameters['test_accuracy'].append(test_accuracy.item())

    print(f'epoch: {epoch} | train_loss: {train_loss:.4f} | train_accuracy: {train_accuracy:.4f} | test_loss: {test_loss:.4f} | test_accuracy: {test_accuracy:.4f}')

  return parameters


