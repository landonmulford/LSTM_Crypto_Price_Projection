
#imports
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from copy import deepcopy as dc

#Load relevant data
data=pd.read_csv('train.csv')
data=data.iloc[100000:]
targetdata=data[['timestamp', 'target']]
closedata=data[['timestamp','close']]


device='cuda:0' if torch.cuda.is_available() else 'cpu'



#Creates dataframe where at each time stamp, contains the 'n' data points leading up to it
#Works for close and target
def prepare_dataframe_for_lstm(df, n, name):

    df = dc(df)
    df.set_index('timestamp', inplace=True)

    for i in range(1, n+1):
        df[name+f'(t-{i})'] = df[name].shift(i)

    df.dropna(inplace=True)

    return df


#Reshapes data to be fit for training, also returns split_index for future reference
def trainReshape(df, indexRatio):
    X=df[:,1:]
    y=df[:,0]
    split_index = int(len(X) * indexRatio)
    X = dc(np.flip(X, axis=1))
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_train = X_train.reshape((-1, lookback))
    return X_train, y_train, split_index


#Reshapes data to be fit for testing
def testReshape(df, indexRatio):
    X = df[:, 1:]
    y = df[:, 0]
    split_index = int(len(X) * indexRatio)
    X = dc(np.flip(X, axis=1))
    X_test = X[split_index:]
    y_test = y[split_index:]
    X_test = X_test.reshape((-1, lookback))
    return X_test, y_test,


#Custom dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

#Defines the model used
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

#Custom Loss Function, more info in ReadMe
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true):
        error = torch.abs(y_pred - y_true)
        for n,_ in enumerate(error[:,1]):
            error[n,1]+=(y_pred[n,1]-0.5)**2*1000

        loss=torch.norm(error)
        return loss.mean()


#Data is split into epochs, this defines training one.
def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 1000 == 999:  # print every 1000 batches
            avg_loss_across_batches = running_loss / 1000
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()


#Runs at the end of an epoch
def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()






if __name__ == '__main__':
    #Creates dataframes and converts them into numpy arrays
    lookback = 4
    close_df = prepare_dataframe_for_lstm(closedata, lookback, 'close')
    target_df=prepare_dataframe_for_lstm(targetdata, lookback, 'target')
    target_df_as_np = target_df.to_numpy()
    close_df_as_np = close_df.to_numpy()


    #splitting data into testing and training, and reshaping
    split_ratio = 0.80
    X_close_train, y_close_train, split_Index = trainReshape(close_df_as_np, split_ratio)
    X_close_test, y_close_test = testReshape(close_df_as_np, split_ratio)
    X_target_train, y_target_train, _ = trainReshape(target_df_as_np, split_ratio)
    X_target_test, y_target_test = testReshape(target_df_as_np, split_ratio)


    #stacking arrays on top of each other, so that both close and target get inputted into the LSTM
    X_train = np.stack((X_close_train, X_target_train), axis=-1)
    y_train = np.stack((y_close_train, y_target_train), axis=-1)
    X_test = np.stack((X_close_test, X_target_test), axis=-1)
    y_test = np.stack((y_close_test, y_target_test), axis=-1)

    #Converting data into PyTorch Tensors
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    #Inputting data into Dataset class
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)


    #Loading data for training and testing
    batch_size = 100
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #Setting LSTM conditions
    model = LSTM(2, 4, 2)
    model.to(device)


    #Setting optimizer conditions
    learning_rate = 0.01
    num_epochs = 1
    loss_function = CustomLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #Running the training
    for epoch in range(num_epochs):
        train_one_epoch()
        validate_one_epoch()


    #Getting output from test data and splitting it
    test_predictions = model(X_test.to(device)).detach().cpu().numpy()


    #Splitting output data into close and target
    close_predictions= test_predictions[:, 0].flatten()
    target_predictions = test_predictions[:, 1].flatten()


    #Getting Results

    # if predicted value for target is above 0.5, it implies a target of 1 and vice versa.
    # This records how many predictions the model gets right
    score=0.0
    for index, value in enumerate(target_predictions):
        if value>0.5 and y_target_test[index]==1.0:
            score+=1
        elif value<=0.5 and y_target_test[index]==0.0:
            score+=1

    percentRight=score/len(target_predictions)*100

    print("To analyze, look at percent of time that the target prediction was right")
    print("The percent right is ", percentRight)



    #This finds the percent error for each individual prediction, and averages them across the whole dataset
    score=0.0
    for index, value in enumerate(close_predictions):
        score+=abs((value-y_close_test[index]) / y_close_test[index])
    averageErrorAsPercent=score / len(close_predictions) * 100
    print("The average percent error of the closes is", averageErrorAsPercent)

