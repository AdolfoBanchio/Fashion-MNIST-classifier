import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

class FashionMNISTClassifier(nn.Module):
  def __init__(self, n1,n2, dropout_val=0.2):
    super(FashionMNISTClassifier, self).__init__()
    self.flatten = nn.Flatten()
    self.full_connected_1 = nn.Linear(28*28,n1)
    self.full_connected_2 = nn.Linear(n1,n2)
    self.output = nn.Linear(n2,10)

    # define activation funtion and dropout
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout_val)

  def forward(self, x):
    # first flatten the input image
    x = self.flatten(x)

    # forward to the first hidden layer
    x = self.relu(self.full_connected_1(x))
    x = self.dropout(x)

    # forward to the second hidden layer
    x = F.relu(self.full_connected_2(x))
    x = self.dropout(x)

    # forward to the output layer
    x = self.output(x)
    return x


def train_batches_fashionMNIST_classifier(model, train_loader, optimizer, criterion):
  """
  Trains the model for all the batches in train_loader

  Returns:
  - avg entropy loss
  - avg accuracy
  """
  model.train() # Se pone el modelo en modo de entrenamiento
  sum_batch_avg_loss = 0 # Inicializamos la suma de las pérdidas promedio de los batches
  sum_correct = 0 # Inicializamos la suma de las predicciones correctas
  num_processed_examples = 0 # Inicializamos la cantidad de ejemplos procesados
  for batch_number, (images, labels) in enumerate(train_loader):
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      images = images.to(device) # Se envía la imagen al dispositivo
      labels = labels.to(device) # Se envía la etiqueta al dispositivo

      batch_size = len(images) # Se obtiene el tamaño del lote
      # Se obtiene la predicción del modelo y se calcula la pérdida 
      pred = model(images)
      loss = criterion(pred, labels)
      
      # Backpropagation usando el optimizador 
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      # Calculamos la perdida promedio del batch y lo agregamos a la suma total
      batch_avg_loss = loss.item() 
      sum_batch_avg_loss += batch_avg_loss
      # Calculamos la cantidad de predicciones correctas
      sum_correct += (pred.argmax(1) == labels).sum().item()
      # Calculamos la cantidad total de predicciones procesadas
      num_processed_examples += batch_size
      # Mostramos el progreso del entrenamiento
  # Calculamos la perdida promedio de todos los batches
  avg_loss = sum_batch_avg_loss / len(train_loader)
  # Calculamos la precisión del modelo
  accuracy = sum_correct / len(train_loader.dataset)
  return avg_loss, accuracy 


def valid_batches_fashionMNIST_classifier(model, valid_loader, criterion):
    model.eval() # Se pone el modelo en modo de evaluación

    sum_batch_avg_loss = 0 # Inicializamos la suma de las pérdidas promedio de los batches
    sum_correct = 0 # Inicializamos la suma de las predicciones correctas
    num_processed_examples = 0 # Inicializamos la cantidad de ejemplos procesados

    for batch_number, (images, labels) in enumerate(valid_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        images = images.to(device) # Se envía la imagen al dispositivo
        labels = labels.to(device) # Se envía la etiqueta al dispositivo
      
        batch_size = len(images)

        # Se obtiene la predicción del modelo y se calcula la pérdida
        pred = model(images)
        loss = criterion(pred, labels)

        # Calculamos la perdida promedio del batch y lo agregamos a la suma total
        batch_avg_loss = loss.item()
        sum_batch_avg_loss += batch_avg_loss

        # Calculamos la cantidad de predicciones correctas
        sum_correct += (pred.argmax(1) == labels).sum().item()
        
        # Calculamos la cantidad total de predicciones procesadas
        num_processed_examples += batch_size

    # Calculamos la perdida promedio de todos los batches
    avg_loss = sum_batch_avg_loss / len(valid_loader)
    # Calculamos la precisión del modelo
    accuracy = sum_correct / len(valid_loader.dataset)

    return avg_loss, accuracy


def train_fashionMNIST_classifier(model, train_loader, valid_loader, optimizer, criterion, epochs):
  train_entropy_loss_incorrect = []
  train_entropy_loss = []
  train_accuracy_incorrect = []
  train_accuracy = []
  
  valid_entropy_loss = []
  valid_accuracy = []
  for epoch in tqdm(range(epochs)):

    # train one epoch
    train_entropy_inc, train_acc_inc = train_batches_fashionMNIST_classifier(model, train_loader, optimizer, criterion)
    train_entropy_loss_incorrect.append(train_entropy_inc)
    train_accuracy_incorrect.append(train_acc_inc)

    # check avg loss and accuracy for incorrect predictions
    train_entropy, train_acc = valid_batches_fashionMNIST_classifier(model, train_loader, criterion)
    train_entropy_loss.append(train_entropy)
    train_accuracy.append(train_acc)

    # validate the epoch
    valid_entropy, valid_acc = valid_batches_fashionMNIST_classifier(model, valid_loader, criterion)
    valid_entropy_loss.append(valid_entropy)
    valid_accuracy.append(valid_acc)

  return train_entropy_loss_incorrect, train_entropy_loss, valid_entropy_loss, train_accuracy_incorrect, train_accuracy, valid_accuracy