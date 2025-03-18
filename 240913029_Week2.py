import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

#%%
torch.manual_seed(42)

np.random.seed(42)

def generate_regression_data(n_sample=1000):
    X=np.random.uniform(-10,10,(n_sample,1))
    y=2*X**2+3+np.random.normal(0,4,(n_sample,1))
    return X,y

#%%

class LinearRegressionModel(nn.Module):
    def __init__(self,input_dim):
        super(LinearRegressionModel,self).__init__()
        self.linear=nn.Linear(input_dim,1)
    def forward(self,x):
        return self.linear(x)

class RegressionNN(nn.Module):
    def __init__(self,input_dim,hidden_dim=64,dropout_rate=0.2):
        super(RegressionNN,self).__init__()
        self.network=nn.Sequential(
            nn.Linear(input_dim,hidden_dim), #input
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim,hidden_dim), #hidden
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim,1) #output
        )

        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self,x):
        return self.network(x)


#%%
def train_regression_model(model,X_Train,y_train,X_val,y_val,epochs=100,lr=0.01,weight_decay=0.0):
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)

    train_losses=[]
    val_losses=[]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs=model(X_Train)
        loss=criterion(outputs,y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())


        model.eval()

        with torch.no_grad():
            val_outputs=model(X_val)
            val_loss=criterion(val_outputs,y_val)
            val_losses.append(val_loss.item())

        if epoch%10==0:
            print(f'Epoch{epoch}: Train Loss:{loss.item():.4f}, val loss={val_loss.item():.4f}')


    return train_losses,val_losses

#%%
def run_regression_example():
    X,y=generate_regression_data()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.fit_transform(X_test)


    X_train_tensor=torch.FloatTensor(X_train_scaled)
    y_train_tensor=torch.FloatTensor(y_train)
    X_test_tensor=torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test)


    print("Training Linear Regression")
    linear_model=LinearRegressionModel(input_dim=1)
    train_losses_linear,val_losses_linear=train_regression_model(
        linear_model,X_train_tensor,y_train_tensor,
        X_test_tensor,y_test_tensor
    )

    print('Training Neural Network')
    nn_model=RegressionNN(input_dim=1)
    train_losses_nn,val_losses_nn=train_regression_model(
        nn_model,X_train_tensor,y_train_tensor,
        X_test_tensor,y_test_tensor,
        weight_decay=0.01
    )

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.scatter(X_train,y_train,alpha=0.5,label='Training Data')
    X_sorted,idx=torch.sort(X_train_tensor,dim=0)
    with torch.no_grad():
        y_pred_linear=linear_model(X_sorted)
        y_pred_nn=nn_model(X_sorted)

    plt.plot(X_sorted.numpy(),y_pred_linear.numpy(),'r-',label='Linear Regression')
    plt.plot(X_sorted.numpy(),y_pred_nn.numpy(),'g-',label='Neural Network')
    plt.legend()
    plt.title('Regression Predictions')

    plt.subplot(1,2,2)
    plt.plot(train_losses_linear,label='linear -train')
    plt.plot(val_losses_linear,label='linear -val')
    plt.plot(train_losses_nn,label='NN -train')
    plt.plot(val_losses_nn,label='NN -val')
    plt.legend()
    plt.title('Loss Curves')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

#%%

print("Running Regression Example:")
run_regression_example()