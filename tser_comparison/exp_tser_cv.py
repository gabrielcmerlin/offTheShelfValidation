import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from aeon.datasets import load_regression
from model_loader import load_models
import scipy.stats as stats
import time
import json

datasets_used = ['AppliancesEnergy', 'HouseholdPowerConsumption1', 'HouseholdPowerConsumption2', 'BenzeneConcentration', 'BeijingPM25Quality']
# datasets_used = ['BeijingPM10Quality', 'FloodModeling1', 'FloodModeling2', 'AustraliaRainfall', 'PPGDalia']
# datasets_used = ['IEEEPPG', 'NewsHeadlineSentiment', 'NewsTitleSentiment', 'Covid3Month']

models_name_list = ['FCN', 'FCN_biggerKernels', 'FCN_moreKernels', 'FCN_moreLayers']

N_EPOCHS = 2000  # Número de épocas
LR = 0.001  # Taxa de aprendizado
SEED = 50
PATIENCE = 100  # Paciencia para early stopping
KFOLDS = 10  # Número de folds para Cross Validation

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda')
print('-----> ', device)

def train(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, save_path, single_result):
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        lr_scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Storing the losses to put it later in a JSON file.
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        single_result['train_losses'] = train_losses
        single_result['val_losses'] = val_losses
        
        print(f"Epoch {epoch+1}/{N_EPOCHS}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

def evaluate(model, test_loader, single_result):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    # Storing the metrics to put it later in a JSON file.
    single_result['mse'] = mse = float(mean_squared_error(y_true, y_pred))
    single_result['rmse'] = np.sqrt(mse)
    single_result['mae'] = float(mean_absolute_error(y_true, y_pred))

if __name__ == '__main__':
    for dataset in datasets_used:
        print(dataset)
        
        for model_name in models_name_list:
            print(model_name)

            X, y = load_regression(dataset)
            X = stats.zscore(X, axis=2) # Normalizing series.
            X = torch.tensor(X).float()
            y = torch.tensor(y).float().unsqueeze(1)

            kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

            results = []
            for i in range(KFOLDS):
                results.append({})

            for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
                start_time = time.time()

                print(f'Fold {fold + 1}/{KFOLDS}')
                
                X_train, X_val, y_train, y_val = train_test_split(X[train_idx], y[train_idx], test_size=0.1, random_state=SEED)
                train_dataset = TensorDataset(X_train, y_train)
                val_dataset = TensorDataset(X_val, y_val)
                test_dataset = Subset(TensorDataset(X, y), test_idx)

                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

                model = load_models(model_name, input_channels=X.shape[1]).to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=LR)
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=0.0001)

                path = f'./checkpoints/tser/{dataset}/{model_name}/'
                save_path = path + f'best_model_fold{fold + 1}.pth'
                train(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, save_path, results[fold])

                model.load_state_dict(torch.load(save_path, weights_only=True))
                evaluate(model, test_loader, results[fold])
                
                # Preparing the training time to store it later.
                end_time = time.time()
                time_spent = np.round(end_time-start_time, 2)
                results[fold]['time'] = time_spent

                # Printing some useful data.
                print('Dataset: ', dataset, '(', fold+1, ')')
                print('Model: ', model_name)
                print('    MAE: ', results[fold]['mae'])
                print('    MSE: ', results[fold]['mse'])
                print('   RMSE: ', results[fold]['rmse'])
                print('   Time: ', results[fold]['time'], 'seconds')
                print('----------------------------------')

            # Compute final MAE across folds
            results_final = {}

            results_final['mae_final'] = mae_final = np.mean([res['mae'] for res in results])
            results_final['std_mae'] = std_mae = np.std([res['mae'] for res in results])
            results_final['mse_final'] = mse_final = np.mean([res['mse'] for res in results])
            results_final['std_mse'] = std_mse = np.std([res['mse'] for res in results])
            results_final['rmse_final'] = rmse_final = np.mean([res['rmse'] for res in results])
            results_final['std_rmse'] = std_rmse = np.std([res['rmse'] for res in results])
            results_final['time_final'] = time_final = np.mean([res['time'] for res in results])
            results_final['std_time'] = std_time = np.std([res['time'] for res in results])
            print('Dataset: ', dataset, '(FINAL)')
            print('Model: ', model_name)
            print(f'    MAE: {mae_final:.4f} ± {std_mae:.4f}')
            print(f'    MSE: {mse_final:.4f} ± {std_mse:.4f}')
            print(f'   RMSE: {rmse_final:.4f} ± {std_rmse:.4f}')
            print(f'   Time: {time_final:.4f} ± {std_time:.4f}')


            # Storing all the data needed.
            with open(path+'resultados_individual.json', 'w') as f:
                json.dump(results, f)

            # Storing all the data needed.
            with open(path+'resultados_aglomerados.json', 'w') as f:
                json.dump(results_final, f)