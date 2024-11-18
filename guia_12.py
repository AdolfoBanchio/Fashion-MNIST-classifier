import fashion_classifier
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda, Compose
#import dill
import json
import os

def train_model(params, t_dataset, v_dataset):
    """ 
    Trains and verifies a fashion_mnist classifier model with the given parameters

    Returns:
    - train_loss
    - train_accuracy
    - valid_loss
    - valid_accuracy
    """
    lr = params['learning_rate']
    dropout = params['dropout']
    n1 = params['n1']
    n2 = params['n2']
    epochs = params['epochs']
    batches_size = params['batches_size']
    train_loader = DataLoader(t_dataset,batch_size=batches_size, shuffle=True, num_workers=os.cpu_count()-1)
    valid_loader = DataLoader(v_dataset,batch_size=batches_size, shuffle=True, num_workers=os.cpu_count()-1)

    # create the model
    model = fashion_classifier.FashionMNISTClassifier(n1,n2,dropout)

    # send model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # set workers to 2
    torch.set_num_threads(2)
    
    # create the optimizer
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # create the loss function
    criterion = nn.CrossEntropyLoss()

    # train the model
    train_loss_inc, train_loss, valid_loss, train_acc_inc, train_acc, valid_acc  = fashion_classifier.train_fashionMNIST_classifier(model, train_loader, valid_loader, optimizer, criterion, epochs)

    return model, train_loss_inc, train_loss, valid_loss, train_acc_inc, train_acc, valid_acc

def train_configurations(configurations, fashion_dataset, validation_dataset):
    # train the 3 different configurations
    for conf_name, conf_params in configurations.items():
        print(f"Training configuration: {conf_name}")
        print(f"Parameters: {conf_params}")
        
        model, train_loss_inc, train_loss, valid_loss, train_acc_inc, train_acc, valid_acc = train_model(conf_params, fashion_dataset, validation_dataset)

        # save the model and the results
        model_path = f"./models/{conf_name}.pt"
        torch.save(model, model_path)
        results = {
            "params": conf_params,
            "train_loss_inc": train_loss_inc,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train_acc_inc": train_acc_inc,
            "train_acc": train_acc,
            "valid_acc": valid_acc
        }
        results_path = f"./results/{conf_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f)

# get the MNIST dataset from pytorch
transformer = transforms.ToTensor()
fashion_dataset = datasets.FashionMNIST(root='./data/fashion/',train=True, download=True, transform=transformer)
validation_dataset = datasets.FashionMNIST(root='./data/fashion/',train=False, download=True, transform=transformer)

# define the parameters for the different configurations
""" "conf_1": { # base configuration
        "learning_rate": 0.001,
        "dropout": 0.2,
        "n1": 128,
        "n2": 64,
        "epochs": 60,
        "batches_size": 100
    },
    "conf_2": { # higher learning rate 
        "learning_rate": 0.01,
        "dropout": 0.2,
        "n1": 128,
        "n2": 64,
        "epochs": 60,
        "batches_size": 100
    },
    "conf_3": { 
        "learning_rate": 0.1,
        "dropout": 0.2,
        "n1": 128,
        "n2": 64,
        "epochs": 60,
        "batches_size": 100
    },
    "conf_4": { 
        "learning_rate": 0.01,
        "dropout": 0.2,
        "n1": 128,
        "n2": 64,
        "epochs": 60,
        "batches_size": 100
    }
    """
configurations = {
    "conf_5": { # base configuration with adam optimizer
        "learning_rate": 0.001,
        "dropout": 0.2,
        "n1": 128,
        "n2": 64,
        "epochs": 60,
        "batches_size": 100
    }
}

train_configurations(configurations, fashion_dataset, validation_dataset)