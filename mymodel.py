import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("Bibliotecas importadas com sucesso.")

# =====================
# TREATED MLP
# =====================
class TreatedMLP:
    def __init__(self,
                 layers=[],
                 treatment=lambda x: torch.empty(x.size(0), 0),
                 num_inputs=2,
                 num_new_inputs=0,
                 num_outputs=1,
                 activation_fn=nn.Tanh,
                 output_activation_fn=nn.Sigmoid,
                 criterion=nn.BCELoss(),
                 learning_rate=0.001,
                 num_batches=1,
                 seed=None):

        self.seed = seed if seed is not None else random.randint(0, 1000000)
        self._set_seed(self.seed)

        self.treatment = treatment
        self.num_inputs = num_inputs
        self.num_new_inputs = num_new_inputs
        self.total_inputs = num_inputs + num_new_inputs
        self.layers = layers
        self.activation_fn = activation_fn
        self.output_activation_fn = output_activation_fn

        self.criterion = criterion
        self.lossType = type(self.criterion).__name__

        self.loss_target_dtype = {
            "CrossEntropyLoss": torch.long,
            "NLLLoss": torch.long,
            "BCELoss": torch.float32,
            "BCEWithLogitsLoss": torch.float32,
            "MSELoss": torch.float32,
            "L1Loss": torch.float32,
            "HuberLoss": torch.float32,
            "KLDivLoss": torch.float32
        }

        self.losses_expect_logits = {"CrossEntropyLoss", "BCEWithLogitsLoss"}

        self.num_outputs = num_outputs
        self.num_batches = num_batches

        self.model = self._build_model()
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.epochs_trained = 0
        self.training_loss = []
        self.test_loss = []

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _build_model(self):
        modules = []
        input_size = self.total_inputs

        if self.layers:
            for neurons in self.layers:
                modules.append(nn.Linear(input_size, neurons))
                modules.append(self.activation_fn())
                input_size = neurons

        modules.append(nn.Linear(input_size, self.num_outputs))

        if self.output_activation_fn is not None and self.lossType not in self.losses_expect_logits:
            modules.append(self.output_activation_fn())

        return nn.Sequential(*modules)

    def train(self, X, Y, X_test=None, Y_test=None, max_epochs=1000, loss_threshold=None):
        X = torch.tensor(X, dtype=torch.float32)
        target_dtype = self.loss_target_dtype.get(self.lossType, torch.float32)
        Y = torch.tensor(Y, dtype=target_dtype)
        if self.lossType in ["CrossEntropyLoss", "NLLLoss"] and Y.dim() > 1:
            Y = Y.view(-1)

        if X_test is not None and Y_test is not None:
            X_test = torch.tensor(X_test, dtype=torch.float32)
            Y_test = torch.tensor(Y_test, dtype=target_dtype)
            if self.lossType in ["CrossEntropyLoss", "NLLLoss"] and Y_test.dim() > 1:
                Y_test = Y_test.view(-1)

        dataset_size = len(X)
        batch_size = max(dataset_size // self.num_batches, 1)

        for epoch in range(max_epochs):
            perm = torch.randperm(dataset_size)
            epoch_loss = 0.0

            for i in range(0, dataset_size, batch_size):
                indices = perm[i:i + batch_size]
                batch_x, batch_y = X[indices], Y[indices]

                if self.num_new_inputs > 0:
                    treated_x = self.treatment(batch_x)
                    if treated_x.size(1) != self.num_new_inputs:
                        raise ValueError(f"A função de tratamento deve retornar {self.num_new_inputs} features, mas retornou {treated_x.size(1)}")
                    batch_x = torch.cat([batch_x, treated_x], dim=1)

                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / (dataset_size / batch_size)
            self.training_loss.append(avg_epoch_loss)

            if X_test is not None and Y_test is not None:
                if self.num_new_inputs > 0:
                    treated_test = self.treatment(X_test)
                    X_test_comb = torch.cat([X_test, treated_test], dim=1)
                else:
                    X_test_comb = X_test

                with torch.no_grad():
                    test_output = self.model(X_test_comb)
                    test_loss_val = self.criterion(test_output, Y_test).item()
                self.test_loss.append(test_loss_val)

            self.epochs_trained += 1
            if loss_threshold is not None and avg_epoch_loss <= loss_threshold:
                print(f"Limiar de perda atingido na época {self.epochs_trained}.")
                break

    def __call__(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if self.num_new_inputs > 0:
            treated = self.treatment(X)
            combined = torch.cat([X, treated], dim=1)
        else:
            combined = X
        with torch.no_grad():
            output = self.model(combined)
        return output.numpy()

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_loss, label="Training Loss")
        if self.test_loss:
            plt.plot(self.test_loss, label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel(f"Loss ({self.lossType})")
        plt.title(f"Perda Durante o Treinamento (Total de {self.epochs_trained} épocas)")
        plt.legend()
        plt.grid(True)
        plt.show()

print("Classe TreatedMLP definida com sucesso.")

# =====================
# Carregar Iris Dataset
# =====================
iris = load_iris()
X, y = iris.data, iris.target

# Normalização
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

num_outputs = len(np.unique(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
num_inputs = X_train.shape[1]

# =====================
# Feature engineering: quadrado das features
# =====================
def squared_treatment(X):
    return X ** 2

# =====================
# Criar o modelo
# =====================
model_iris = TreatedMLP(
    layers=[16, 8],
    treatment=squared_treatment,
    num_inputs=num_inputs,
    num_new_inputs=num_inputs,
    num_outputs=num_outputs,
    activation_fn=torch.nn.ReLU,
    output_activation_fn=None,  # CrossEntropyLoss faz softmax interno
    criterion=torch.nn.CrossEntropyLoss(),
    learning_rate=0.001,
    num_batches=4
)

# =====================
# Treinar o modelo
# =====================
model_iris.train(X_train, y_train, X_test, y_test, max_epochs=1000)
model_iris.plot_loss()

# =====================
# Função para classificar pelo console
# =====================
def classify_flower():
    print("\nDigite as medidas da flor (em cm):")
    sepal_length = float(input("Sepal length: "))
    sepal_width  = float(input("Sepal width: "))
    petal_length = float(input("Petal length: "))
    petal_width  = float(input("Petal width: "))

    x = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    x = scaler.transform(x)  # usa mesmo scaler do treino

    output = model_iris(x)
    pred_class = np.argmax(output, axis=1)[0]

    iris_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
    print(f"\nPredição do modelo: {iris_names[pred_class]}")

# Função para avaliar o modelo
def avaliar_modelo(modelo, X_test, y_test):
    # Realiza as previsões
    previsoes = modelo(X_test)
    # Converte as previsões para rótulos de classe
    previsoes_classes = np.argmax(previsoes, axis=1)
    # Calcula a precisão
    precisao = accuracy_score(y_test, previsoes_classes)
    return precisao

# Avalia o modelo
precisao = avaliar_modelo(model_iris, X_test, y_test)
print(f"Acurácia no conjunto de teste: {precisao:.4f}")
# =====================
# Rodar Programa
# =====================

while 1:
    escolha = int(input("Escolha(1 ou 2) -- 1 Rodar Modelo -- 2 Encerrar"))
    if escolha == 1:
        classify_flower()
    else:
        break
