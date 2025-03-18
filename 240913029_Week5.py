'''
[INFO] [INFO] [INFO]
In this exercise you will
1) Implement a neural network of your specifications in My_Reg class and My_Clf class
2) Decide the training parameters using regularization for training both models
4) Plot the curves to check performance
3) Create a multi class classification dataset
4) New concepts introduced
    - Hybrid Regularization consisting of L1 and L2 regularizers
    - Multiclass classification
    - MCC, Micro and Macro F1  performance quantifier for multiclass classification
    - Adagrad and Stochastic GD

[PRO CHALLENGE] (OPTIONAL)
Once you run the models you will observe that classification is easy task
However the R2 and Adjusted R2 are not that good try to improve these metrics
by changing your model.

[PRO MAX CHALLENGE] (OPTIONAL)
For multiclass classification all metrics are poor and the model needs significant
improvments. Give it a try.
[INFO] [INFO] [INFO]
'''

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, f1_score, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt


class My_Reg(nn.Module):
    def __init__(self, input_dim):
        super(My_Reg, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

class My_Clf(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(My_Clf, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def train_regression_model(X, y, learning_rate='0.01', epochs='200'):
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).reshape(-1, 1)
    model = My_Reg(X.shape[1])

    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer_adagrad = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        optimizer = optimizer_sgd if epoch % 2 == 0 else optimizer_adagrad
        outputs = model(X)
        loss = criterion(outputs, y)

        l1_lambda = 0.01  # Try changing this value and experimenting
        l2_lambda = 0.01  # Try changing this value and experimenting
        l1_reg = torch.tensor(0.)
        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            l1_reg += torch.norm(param, 1)
            l2_reg += torch.norm(param, 2)

        loss = loss + l1_lambda * l1_reg + l2_lambda * l2_reg

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return model, losses


def train_classification_model(X, y, num_classes=2, learning_rate=0.01, epochs=200):
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)

    model = My_Clf(X.shape[1], num_classes)
    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer_adagrad = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        optimizer = optimizer_sgd if epoch % 2 == 0 else optimizer_adagrad

        outputs = model(X)
        loss = criterion(outputs, y)

        l1_lambda = 0.01
        l2_lambda = 0.01
        l1_reg = torch.tensor(0.)
        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            l1_reg += torch.norm(param, 1)
            l2_reg += torch.norm(param, 2)

        loss = loss + l1_lambda * l1_reg + l2_lambda * l2_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, losses


def evaluate_regression(model, X, y):
    X = torch.FloatTensor(X)
    predictions = model(X).detach().numpy()
    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    n = len(y)
    p = X.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return {
        'R2': r2,
        'RMSE': rmse,
        'Adjusted R2': adjusted_r2
    }


def evaluate_classification(model, X, y, num_classes=2):
    X = torch.FloatTensor(X)
    outputs = model(X)
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.numpy()
    if num_classes == 2:
        f1 = f1_score(y, predicted)
        conf_matrix = confusion_matrix(y, predicted)
        return {
            'F1 Score': f1,
            'Confusion Matrix': conf_matrix
        }
    else:
        micro_f1 = f1_score(y, predicted, average='micro')
        macro_f1 = f1_score(y, predicted, average='macro')
        mcc = matthews_corrcoef(y, predicted)
        return {
            'Micro F1': micro_f1,
            'Macro F1': macro_f1,
            'MCC': mcc
        }


california = fetch_california_housing()
X_reg, y_reg = california.data, california.target

cancer = load_breast_cancer()
X_class, y_class = cancer.data, cancer.target

# Create multiclass dataset
X_multi, y_multi = make_classification(n_samples=1000, n_features=20,
                                       n_informative=15, n_redundant=5,
                                       n_classes=4, random_state=42)

# Split datasets
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2)
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class, y_class, test_size=0.2)
X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(X_multi, y_multi, test_size=0.2)

scaler = StandardScaler()
# Scaling the datasets
X_reg_train = scaler.fit_transform(X_reg_train)
X_reg_test = scaler.transform(X_reg_test)

X_class_train = scaler.fit_transform(X_class_train)
X_class_test = scaler.transform(X_class_test)

X_multi_train = scaler.fit_transform(X_multi_train)
X_multi_test = scaler.transform(X_multi_test)

# Train Regression Model
reg_model, reg_losses = train_regression_model(X_reg_train, y_reg_train, learning_rate=0.01, epochs=200)
reg_metrics = evaluate_regression(reg_model, X_reg_test, y_reg_test)

# Train Binary Classification Model
class_model, class_losses = train_classification_model(X_class_train, y_class_train, num_classes=2, learning_rate=0.01, epochs=200)
class_metrics = evaluate_classification(class_model, X_class_test, y_class_test, num_classes=2)

# Train Multiclass Classification Model
multi_class_model, multi_losses = train_classification_model(X_multi_train, y_multi_train, num_classes=4, learning_rate=0.01, epochs=200)
multi_metrics = evaluate_classification(multi_class_model, X_multi_test, y_multi_test, num_classes=4)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(reg_losses)
plt.title('Regression Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1, 3, 2)
plt.plot(class_losses)
plt.title('Binary Classification Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 3, 3)
plt.plot(multi_losses)
plt.title('Multiclass Classification Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

print("\nRegression Metrics:")
for metric, value in reg_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nBinary Classification Metrics:")
print(f"F1 Score: {class_metrics['F1 Score']:.4f}")
print("Confusion Matrix:")
print(class_metrics['Confusion Matrix'])

print("\nMulticlass Classification Metrics:")
for metric, value in multi_metrics.items():
    print(f"{metric}: {value:.4f}")