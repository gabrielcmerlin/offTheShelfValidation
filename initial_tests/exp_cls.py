import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from aeon.datasets import load_classification

import time
import json

# Models using Batch Normalization.
# from models.fcn import FCNClassifier
# from models.fcn_biggerKernels import FCNClassifier
# from models.fcn_moreKernels import FCNClassifier
from models.fcn_moreLayers import FCNClassifier

N_EXP    = 5
N_EPOCHS = 1000
PATIENCE = 20
LR       = 0.001 # Leaning rate

# Univariate + multivariate datasets.
# datasets_used = ['ArrowHead', 'OSULeaf'] + ['NATOPS', 'Epilepsy'] # First experiment (general purpose)
# datasets_used = ['Haptics'] # Second experiment (more complex dataset)
datasets_used = ['ArrowHead', 'NATOPS', 'Haptics'] # Third experiment

device = torch.device('cuda')
print('-----> ', device)

# Training the model.
def train(model, train_loader, val_loader, criterion, optimizer, save_path, single_result):
    # Inicializing the best validation loss to do a early stopping.
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []
    
    for epoch in range(N_EPOCHS):
        # Training loop.
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Moving data to GPU.

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Moving data to GPU.

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Verifying a possible early stopping.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                torch.load(save_path, weights_only=True)
                break
        
        # Storing the losses to put it later in a JSON file.
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        single_result['train_losses'] = train_losses
        single_result['val_losses'] = val_losses
        
        print(f"Epoch {epoch+1}/{N_EPOCHS}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

# Evaluating the model.
def evaluate(model, test_loader, single_result):
    model.eval()

    y_true = []
    y_pred = []

    # Evaluation loop.
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Moving data to GPU.

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    # Storing the metrics and labels to put it later in a JSON file.
    single_result['acc'] = accuracy_score(y_true, y_pred)
    single_result['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    single_result['f1_micro'] = f1_score(y_true, y_pred, average='micro')
    single_result['y_true'] = [int(x) for x in y_true]
    single_result['y_pred'] = [int(x) for x in y_pred]

if __name__ == '__main__':
    for dataset in datasets_used:
        # Loading the dataset.
        X_train, y_train = load_classification(dataset, split="train")
        X_test, y_test = load_classification(dataset, split="test")

        # Changing the labels (strings) to intengers.
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        # Converting the data to PyTorch tensors.
        X_train = torch.tensor(X_train).float().to(device)
        y_train = torch.tensor(y_train).long().to(device)
        X_test = torch.tensor(X_test).float().to(device)
        y_test = torch.tensor(y_test).long().to(device)

        # Dividing train data into train and validation parts.
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Creating DataLoader to load the data in batches.
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Storing some data (metrics, losses and labels) to put it later into a JSON file.
        results = []
        for i in range(N_EXP):
            results.append({})

        for i in range(N_EXP):
            start_time = time.time()

            # Defining the model, loss function and optimizer.
            model = FCNClassifier(input_channels=X_train.shape[1], num_classes=len(torch.unique(y_train))).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LR)

            # Changing the path where the data will be saved.
            save_path = './checkpoints/cls/' + dataset + '/FCN_moreLayers/'
            file_name = 'best_model_' + str(i+1) + '.pth'

            # Training and evaluating the model.
            train(model, train_loader, val_loader, criterion, optimizer, save_path+file_name, results[i])
            evaluate(model, test_loader, results[i])

            # Preparing the training time to store it later.
            end_time = time.time()
            time_spent = np.round(end_time-start_time, 2)
            results[i]['time'] = time_spent

            # Printing some useful data.
            print(' Dataset: ', dataset, '(', i+1, ')')
            print('Acurácia: ', results[i]['acc'])
            print('F1 Macro: ', results[i]['f1_macro'])
            print('F1 Micro: ', results[i]['f1_micro'])
            print('    Time: ', results[i]['time'], 'seconds')
            print('----------------------------------')

        # Storing all the data needed.
        with open(save_path+'data.json', 'w') as f:
            json.dump(results, f)