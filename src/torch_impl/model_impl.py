from sklearn.preprocessing import StandardScaler
from src.utils.get_data import get_data
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import torch.nn as nn
import torch.optim as optim
from time import time
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import pandas as pd
from src.utils.get_data import get_scaler

df_list = get_data()
print("get_data successeful")

class RegModel():
    """
        implementation of a regression model with torch

    """
    def __init__(self,df_list = get_data(),hl1 = 64,hl2 = 32,hl3 = 16,do = 0.2,lr = 0.001,prune_amount = 0.3,nb_epoch = 28):
        # we load the data
        self.X_train = df_list[0]
        self.X_test = df_list[1]
        self.Y_train = df_list[2]
        self.Y_test = df_list[3]

        # we turn it into tensors (object that used by torch)
        self.X_train_tensor = torch.tensor(self.X_train.values,dtype=torch.float32)
        self.Y_train_tensor = torch.tensor(self.Y_train.values,dtype=torch.float32)
        self.X_test_tensor = torch.tensor(self.X_test.values,dtype=torch.float32)
        self.Y_test_tensor = torch.tensor(self.Y_test.values,dtype=torch.float32)


        self.hl1 = hl1
        self.hl2 = hl2
        self.hl3 = hl3

        self.do = do
        self.lr = lr
        self.prune_amount = prune_amount
        self.nb_epoch = nb_epoch

        self.model = nn.Sequential(
            nn.Dropout(self.do),
            nn.Linear(self.X_train_tensor.shape[1],self.hl1),
            nn.ReLU(),
            nn.Dropout(self.do),
            nn.Linear(self.hl1,self.hl2),
            nn.ReLU(),
            nn.Dropout(self.do),
            nn.Linear(self.hl2,self.hl3),
            nn.ReLU(),
            nn.Linear(self.hl3,self.Y_train_tensor.shape[1])
        )
    def train(self):
        start_time = time()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.nb_epoch):
            optimizer.zero_grad()
            outputs = self.model(self.X_train_tensor)
            self.loss = criterion(outputs, self.Y_train_tensor)
            self.loss.backward()
            optimizer.step()

        self.training_time = time() - start_time

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.prune_amount)
                prune.remove(module, 'weight')

    def guess(self):
        start_time = time()
        #y_pred_tensor = torch.argmax(model(X_test_tensor))
        #y_pred = y_pred_tensor.detach().cpu().numpy()
        self.Y_pred_train = self.model(self.X_train_tensor).detach().cpu().numpy()
        self.Y_pred_test = self.model(self.X_test_tensor).detach().cpu().numpy()
        scaler = get_scaler()
        self.Y_pred_train = scaler.inverse_transform(self.Y_pred_train)
        self.Y_pred_test = scaler.inverse_transform(self.Y_pred_test)
        self.Y_pred_test = pd.DataFrame({'ct_money': self.Y_pred_test[:, 0], 'ct_health': self.Y_pred_test[:, 1]})
        self.Y_pred_train = pd.DataFrame({'ct_money': self.Y_pred_train[:, 0], 'ct_health': self.Y_pred_train[:, 1]})
        self.testing_time = time() - start_time

    def get_metrics(self):
        # metrics
        self.y_train_health = self.Y_train["ct_health"]
        self.y_train_money = self.Y_train["ct_money"]
        self.y_test_health = self.Y_test["ct_health"]
        self.y_test_money = self.Y_test["ct_money"]
        self.y_pred_train_health = self.Y_pred_train["ct_health"]
        self.y_pred_train_money = self.Y_pred_train["ct_money"]
        self.y_pred_test_health = self.Y_pred_test["ct_health"]
        self.y_pred_test_money = self.Y_pred_test["ct_money"]

        self.MAE_train_health = mean_absolute_error(self.y_train_health,self.y_pred_train_health)
        self.MAE_train_money = mean_absolute_error(self.y_train_money,self.y_pred_train_money)

        self.MAE_test_health = mean_absolute_error(self.y_test_health,self.y_pred_test_health)
        self.MAE_test_money = mean_absolute_error(self.y_test_money,self.y_pred_test_money)

        self.RMSE_train_health = root_mean_squared_error(self.y_train_health,self.y_pred_train_health)
        self.RMSE_train_money = root_mean_squared_error(self.y_train_money,self.y_pred_train_money)

        self.RMSE_test_health = root_mean_squared_error(self.y_test_health,self.y_pred_test_health)
        self.RMSE_test_money = root_mean_squared_error(self.y_test_money,self.y_pred_test_money)

        self.R2_train = r2_score(self.Y_train_tensor,self.Y_pred_train)
        self.R2_test = r2_score(self.Y_test_tensor,self.Y_pred_test)

        # display metrics
        print(f"---MLP regression (prunned {self.prune_amount}) metrics---\n* Training time (s): {self.training_time}\n* Predicting time (s): {self.testing_time}")
        print(f"* MAE (training - health) : {self.MAE_train_health}\n* MAE (prediction - health) : {self.MAE_test_health}")
        print(f"* MAE (training - money) : {self.MAE_train_money}\n* MAE (prediction - money) : {self.MAE_test_money}")
        print(f"* RMSE (training - health) : {self.RMSE_train_health}\n* RMSE (prediction - health) : {self.RMSE_test_health}")
        print(f"* RMSE (training - money) : {self.RMSE_train_money}\n* RMSE (prediction - money) : {self.RMSE_test_money}")
        print(f"* R2 (training) : {self.R2_train}\n* R2 (prediction) : {self.R2_test}")



class CatModel():
    """
        implementation of a classificatioin model with torch

    """
    def __init__(self,df_list = get_data(),hl1 = 64,hl2 = 32,hl3 = 16,do = 0.2,lr = 0.001,prune_amount = 0.3,nb_epoch = 28):
        # we load the data
        self.X_train = df_list[0]
        self.X_test = df_list[1]
        self.y_train = df_list[4]
        self.y_test = df_list[5]

        # we turn it into tensors (object that used by torch)
        self.X_train_tensor = torch.tensor(self.X_train.values,dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self.y_train.values,dtype=torch.float32)
        self.X_test_tensor = torch.tensor(self.X_test.values,dtype=torch.float32)
        self.y_test_tensor = torch.tensor(self.y_test.values,dtype=torch.float32)


        self.hl1 = hl1
        self.hl2 = hl2
        self.hl3 = hl3

        self.do = do
        self.lr = lr
        self.prune_amount = prune_amount
        self.nb_epoch = nb_epoch

        self.model = nn.Sequential(
            nn.Dropout(self.do),
            nn.Linear(self.X_train_tensor.shape[1],self.hl1),
            nn.ReLU(),
            nn.Dropout(self.do),
            nn.Linear(self.hl1,self.hl2),
            nn.ReLU(),
            nn.Dropout(self.do),
            nn.Linear(self.hl2,self.hl3),
            nn.ReLU(),
            nn.Linear(self.hl3,self.y_train_tensor.shape[1])
        )
    def train(self):
        start_time = time()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.nb_epoch):
            optimizer.zero_grad()
            outputs = self.model(self.X_train_tensor)
            self.loss = criterion(outputs, self.y_train_tensor)
            self.loss.backward()
            optimizer.step()

        self.training_time = time() - start_time

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.prune_amount)
                prune.remove(module, 'weight')

    def guess(self):
        start_time = time()
        y_pred_tensor_train = torch.where(self.model(self.X_train_tensor)>=0.5,1.0,0.0)
        y_pred_tensor_test = torch.where(self.model(self.X_test_tensor)>=0.5,1.0,0.0)
        self.y_pred_train = y_pred_tensor_train.detach().cpu().numpy()
        self.y_pred_test = y_pred_tensor_test.detach().cpu().numpy()
        self.Y_pred_test = pd.DataFrame({'round_winner': self.y_pred_test[:, 0]})
        self.Y_pred_train = pd.DataFrame({'round_winner': self.y_pred_train[:, 0]})
        self.testing_time = time() - start_time

    def get_metrics(self):
        # metrics
        self.acc_train = (self.y_pred_train == self.y_train).sum().item() / self.y_train_tensor.size(0)
        self.acc_test = (self.y_pred_test == self.y_test).sum().item() / self.y_test_tensor.size(0)

        # display metrics
        print(f"---MLP classification (prunned {self.prune_amount}) metrics---\n* Training time (s): {self.training_time}\n* Predicting time (s): {self.testing_time}")
        print(f"* accuracy (training) : {self.acc_train}\n* accuracy (prediction) : {self.acc_test}")


""" regression exemple
reg_model_base = RegModel()
reg_model_base.train()
reg_model_base.guess()
reg_model_base.get_metrics()
"""
""" classification exemple
cat_model_base = CatModel()
cat_model_base.train()
cat_model_base.guess()
cat_model_base.get_metrics()
"""
