import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score


def generate_regression_data(n_samples=1000, n_features=10):
    X = np.random.rand(n_samples, n_features)
    y = np.sum(X, axis=1) + np.random.normal(0, 0.1, size=n_samples)  # y = sum of features + noise
    return train_test_split(X, y, test_size=0.2, random_state=42)

def generate_classification_data(n_samples=1000, n_features=10, n_classes=3):
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, n_classes, size=n_samples)
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = generate_regression_data()
X_train_cls, X_test_cls, y_train_cls, y_test_cls = generate_classification_data()


scaler_reg = StandardScaler().fit(X_train_reg)
X_train_reg, X_test_reg = scaler_reg.transform(X_train_reg), scaler_reg.transform(X_test_reg)

scaler_cls = StandardScaler().fit(X_train_cls)
X_train_cls, X_test_cls = scaler_cls.transform(X_train_cls), scaler_cls.transform(X_test_cls)


X_train_reg, y_train_reg = torch.tensor(X_train_reg, dtype=torch.float32), torch.tensor(y_train_reg, dtype=torch.float32)
X_test_reg, y_test_reg = torch.tensor(X_test_reg, dtype=torch.float32), torch.tensor(y_test_reg, dtype=torch.float32)

X_train_cls, y_train_cls = torch.tensor(X_train_cls, dtype=torch.float32), torch.tensor(y_train_cls, dtype=torch.long)
X_test_cls, y_test_cls = torch.tensor(X_test_cls, dtype=torch.float32), torch.tensor(y_test_cls, dtype=torch.long)

class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(RegressionModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(ClassificationModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 3))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def objective_regression(trial):
    # Suggest hyperparameters
    num_layers = trial.suggest_int("num_layers", 1, 3)
    hidden_units = [trial.suggest_int(f"units_{i}", 32, 128, step=32) for i in range(num_layers)]
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)

    model = RegressionModel(X_train_reg.shape[1], hidden_units)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(20):
        optimizer.zero_grad()
        predictions = model(X_train_reg).squeeze()
        loss = criterion(predictions, y_train_reg)
        loss.backward()
        optimizer.step()

    # Validation loss
    with torch.no_grad():
        y_pred = model(X_test_reg).squeeze().numpy()
    return mean_squared_error(y_test_reg.numpy(), y_pred)

def objective_classification(trial):
    num_layers = trial.suggest_int("num_layers", 1, 3)
    hidden_units = [trial.suggest_int(f"units_{i}", 32, 128, step=32) for i in range(num_layers)]
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)

    model = ClassificationModel(X_train_cls.shape[1], hidden_units, dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(20):
        optimizer.zero_grad()
        predictions = model(X_train_cls)
        loss = criterion(predictions, y_train_cls)
        loss.backward()
        optimizer.step()

    # Validation accuracy
    with torch.no_grad():
        y_pred = torch.argmax(model(X_test_cls), dim=1).numpy()
    return accuracy_score(y_test_cls.numpy(), y_pred)

# Run hyperparameter tuning
study_reg = optuna.create_study(direction="minimize")
study_reg.optimize(objective_regression, n_trials=20)
best_params_reg = study_reg.best_params

study_cls = optuna.create_study(direction="maximize")
study_cls.optimize(objective_classification, n_trials=20)
best_params_cls = study_cls.best_params

# Train best regression model
best_reg_model = RegressionModel(X_train_reg.shape[1], [best_params_reg[f"units_{i}"] for i in range(best_params_reg["num_layers"])])
optimizer = optim.Adam(best_reg_model.parameters(), lr=best_params_reg["learning_rate"])
criterion = nn.MSELoss()

for epoch in range(50):  # Train for more epochs
    optimizer.zero_grad()
    predictions = best_reg_model(X_train_reg).squeeze()
    loss = criterion(predictions, y_train_reg)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    y_pred_reg = best_reg_model(X_test_reg).squeeze().numpy()
mse = mean_squared_error(y_test_reg.numpy(), y_pred_reg)
print(f"Best Regression Model MSE: {mse:.4f}")

# Train best classification model
best_cls_model = ClassificationModel(X_train_cls.shape[1],
                                     [best_params_cls[f"units_{i}"] for i in range(best_params_cls["num_layers"])],
                                     best_params_cls["dropout_rate"])
optimizer = optim.Adam(best_cls_model.parameters(), lr=best_params_cls["learning_rate"])
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    optimizer.zero_grad()
    predictions = best_cls_model(X_train_cls)
    loss = criterion(predictions, y_train_cls)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    y_pred_cls = torch.argmax(best_cls_model(X_test_cls), dim=1).numpy()
accuracy = accuracy_score(y_test_cls.numpy(), y_pred_cls)
print(f"Best Classification Model Accuracy: {accuracy:.4f}")