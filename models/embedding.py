import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os

dataset = pd.read_csv('data_utc/Seattle/data_utc.csv')
mask = (dataset['Year'] != 2021) & (dataset['Year'] != 2022)
train_dataset = dataset[mask].copy()
train_dataset.reset_index(drop=True, inplace=True)

x_train = train_dataset[['Cloud Type']].to_numpy().flatten()
y_train = train_dataset[['GHI']].to_numpy().flatten()
x_train = torch.IntTensor(x_train)
y_train = torch.FloatTensor(y_train)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# x_train = x_train.to(device)
# y_train = y_train.to(device)

embedding_size = 2
model = nn.Sequential(
    nn.Embedding(10, embedding_size),
    nn.Flatten(),
    nn.Linear(embedding_size, 50),
    nn.ReLU(),
    nn.Linear(50, 15),
    nn.ReLU(),
    nn.Linear(15, 1)
)
# model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
epochs = 100

for epoch in range(epochs):
    y_pred = model.forward(x_train)
    loss = criterion(y_pred.squeeze(), y_train)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

layers = list(model.parameters())
print(layers[0])