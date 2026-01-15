import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from aeon.datasets import load_regression

from model_loader import load_models

import time
import json

# Univariate + multivariate datasets.
datasets_used = ['Covid3Month', 'FloodModeling1', 'AppliancesEnergy', 'HouseholdPowerConsumption2', 'IEEEPPG']

# Models using Batch Normalization.
models_name_list = ['FCN', 'FCN_biggerKernels', 'FCN_moreKernels', 'FCN_moreLayers']
N_EXP    = 5
N_EPOCHS = 1000
PATIENCE = 20
LR       = 0.001 # Leaning rate
SEED     = 50

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Moving data to GPU.

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Moving data to GPU.

                outputs = model(inputs)
                loss = criterion(outputs, targets)
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
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Moving data to GPU.

            outputs = model(inputs)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
            
    # Storing the metrics to put it later in a JSON file.
    single_result['mse'] = mse = float(mean_squared_error(y_true, y_pred))
    single_result['rmse'] = np.sqrt(mse)
    single_result['mae'] = float(mean_absolute_error(y_true, y_pred))

if __name__ == '__main__':
    
    for model_name in models_name_list:

        print(model_name)

        for dataset in datasets_used:
            # Loading the dataset.
            X_train, y_train = load_regression(dataset, split="train")
            X_test, y_test = load_regression(dataset, split="test")

            # Converting the data to PyTorch tensors.
            X_train = torch.tensor(X_train).float().to(device)
            y_train = torch.tensor(y_train).float().unsqueeze(1).to(device)
            X_test = torch.tensor(X_test).float().to(device)
            y_test = torch.tensor(y_test).float().unsqueeze(1).to(device)

            # Dividing train data into train and validation parts.
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            # Creating DataLoader to load the data in batches.
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            # Storing some data (metrics, losses and targets) to put it later into a JSON file.
            results = []
            for i in range(N_EXP):
                results.append({})

            for i in range(N_EXP):
                start_time = time.time()

                # Defining the model, loss function and optimizer.
                model = load_models(model_name, input_channels=X_train.shape[1]).to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=LR)

                # Changing the path where the data will be saved.
                save_path = './checkpoints/tser/' + dataset + '/' + model_name + '/'
                file_name = 'best_model_' + str(i+1) + '.pth'

                # Training and evaluating the model.
                train(model, train_loader, val_loader, criterion, optimizer, save_path+file_name, results[i])
                evaluate(model, test_loader, results[i])

                # Preparing the training time to store it later.
                end_time = time.time()
                time_spent = np.round(end_time-start_time, 2)
                results[i]['time'] = time_spent

                # Printing some useful data.
                print('Dataset: ', dataset, '(', i+1, ')')
                print('    MAE: ', results[i]['mae'])
                print('    MSE: ', results[i]['mse'])
                print('   RMSE: ', results[i]['rmse'])
                print('   Time: ', results[i]['time'], 'seconds')
                print('----------------------------------')

            # Storing all the data needed.
            with open(save_path+'data.json', 'w') as f:
                json.dump(results, f)