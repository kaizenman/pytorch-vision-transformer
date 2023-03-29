import matplotlib.pyplot as plt
from pathlib import Path

def visualize_learning(parameters, model_name):
  epochs = range(len(parameters['train_loss']))
  
  f = plt.figure(figsize=(12, 9))
  plt.subplot(2, 2, 1)

  plt.plot(epochs, parameters['train_loss'], label='Train Loss')
  plt.plot(epochs, parameters['test_loss'], label='Test Loss')
  plt.title('Loss')
  plt.xlabel(xlabel='Epochs')
  plt.legend()

  plt.subplot(2, 2, 2)
  plt.plot(epochs, parameters['train_accuracy'], label='Train Accuracy')
  plt.plot(epochs, parameters['test_accuracy'], label='Test Accuracy')
  plt.title('Accuracy')
  plt.xlabel(xlabel='Epochs')
  plt.legend()

  results_dir = Path('results')
  results_image_name = model_name + '.png'
  results_image = results_dir / results_image_name 

  if not results_dir.is_dir():
    results_dir.mkdir(parents=True, exist_ok=True)

  f.savefig(results_image)
  plt.close(f)
