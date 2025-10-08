TreatedMLP - MLP Customizado com Feature Engineering

Descrição:
Este projeto implementa um MLP (Multi-Layer Perceptron) customizado em PyTorch, com suporte a feature engineering e interface de predição via console. O modelo foi testado com o dataset Iris, mas é extensível para outros datasets de classificação ou regressão.

Funcionalidades:
- Definição de arquitetura customizável (camadas, número de neurônios, funções de ativação)
- Suporte a tratamento de features (geração de novas features derivadas das originais)
- Treinamento com mini-batches e seed para reprodutibilidade
- Suporte a várias funções de perda (CrossEntropyLoss, MSELoss, BCELoss, etc.)
- Plot de curvas de perda de treino e teste
- Interface para classificação de novas amostras via console
- Extensível para outros datasets e problemas

Requisitos:
- Python >= 3.10
- PyTorch >= 2.1.0
- scikit-learn >= 1.3.0
- numpy >= 1.25.0
- matplotlib >= 3.8.0

Como usar:
1. Instalar dependências:
   pip install -r requirements.txt

2. Treinar o modelo:
   - Importar bibliotecas:
     from sklearn.datasets import load_iris
     from sklearn.model_selection import train_test_split
     from sklearn.preprocessing import StandardScaler
     import mymodel
   - Carregar e normalizar dataset
   - Definir função de feature engineering
   - Criar modelo e treinar
   - Plotar perda com model.plot_loss()

3. Predição de novas flores:
   - Inserir medidas da flor
   - Normalizar com o mesmo scaler
   - Chamar model(x) para obter previsão
   - Interpretar a classe com {0:"setosa",1:"versicolor",2:"virginica"}

Resultados:
- Acurácia no conjunto de teste: 100%
- O modelo separa perfeitamente as classes do dataset Iris usando MLP + feature engineering

Possíveis melhorias:
- Implementar regularização (dropout, L2)
- Adicionar métricas adicionais (precision, recall, F1)
- Testar em datasets maiores e mais complexos
- Criar interface gráfica para entrada de dados

Autor:
Vitor Quirino - projeto criado como experimento de MLP customizado em PyTorch

