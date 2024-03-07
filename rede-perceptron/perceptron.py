import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt

# Função de ativação degrau
def step_function(x):
    return 1 if x >= 0 else 0

# Inicialização dos pesos e bias pseudo-aleatoriamente
np.random.seed(42)  # Para reprodutibilidade
weights = np.random.rand(3)
learning_rate = float(input("Digite a taxa de aprendizagem: "))

# Carregar dados do arquivo xlsx
file_path = "banco.xlsx"
data = pd.read_excel(file_path)
data.insert(0, 'x0', -1)  # Adiciona coluna x0 com valor -1

# Transformar os dados em uma lista e embaralhar
data_list = data.values.tolist()
random.shuffle(data_list)

# Definir percentual de amostras para treinamento e teste
train_percentage = float(input("Digite o percentual de amostras para treinamento (0-1): "))
train_size = int(train_percentage * len(data_list))
train_data = data_list[:train_size]
test_data = data_list[train_size:]

def plotar_graficos_antes_depois(vetor_pesos, titulo, data_list):
    # Obter todo conjunto de dados como matriz
    matriz_conjunto = np.array(data_list)

    # Extrair as entradas da matriz
    x_1 = matriz_conjunto[:, 1]  # Pressão
    x_2 = matriz_conjunto[:, 2]  # Temperatura

    # Filtrar os pontos com valor 0 e 1
    x_zero = x_1[matriz_conjunto[:, 3] == 0]
    y_zero = x_2[matriz_conjunto[:, 3] == 0]

    x_um = x_1[matriz_conjunto[:, 3] == 1]
    y_um = x_2[matriz_conjunto[:, 3] == 1]

    # Plotar os pontos
    plt.scatter(x_zero, y_zero, marker='o', label='0')
    plt.scatter(x_um, y_um, marker='x', label='1')

    # Calcular os coeficientes da reta
    a = -vetor_pesos[1] / vetor_pesos[2]
    b = vetor_pesos[0] / vetor_pesos[2]

    # Definir os pontos para a reta
    x_values = np.linspace(min(x_1), max(x_1), 100)
    y_values = a * x_values + b
    # Plotar a reta
    plt.plot(x_values, y_values, color='red', linestyle='-', label='Reta separadora')

    plt.xlabel('Pressão')
    plt.ylabel('Temperatura')
    plt.title(titulo)

    plt.legend()
    plt.grid(True)
    plt.show()

plotar_graficos_antes_depois(weights, 'Antes do Treinamento', train_data)

# Treinamento
start_time = time.time()
epochs = 0

while True:
    error_count = 0

    for row in train_data:
        x = np.array(row[:-1])
        y = row[-1]

        # Calcular saída
        output = step_function(np.dot(weights, x))

        # Atualizar pesos apenas se houver erro
        if y != output:
            weights += learning_rate * (y - output) * x
            error_count += 1

    epochs += 1

    # Critério de parada
    if error_count == 0 or epochs > 1000:
        break

training_time = time.time() - start_time
print(f"Tempo de treinamento: {training_time:.4f} segundos")
print(f"Número de épocas necessárias: {epochs}")

plotar_graficos_antes_depois(weights, 'Depois do Treinamento', test_data)


# Teste
start_time = time.time()
correct_count = 0
confusion_matrix = np.zeros((2, 2), dtype=int)

for row in test_data:
    x = np.array(row[:-1])
    y = row[-1]

    # Calcular saída
    output = step_function(np.dot(weights, x))

    # Atualizar matriz de confusão
    confusion_matrix[int(y)][output] += 1

    # Contar acertos
    correct_count += int(y == output)

testing_time = time.time() - start_time
print(f"Tempo de teste: {testing_time:.4f} segundos")
print(f"Percentual de acerto: {correct_count / len(test_data) * 100:.2f}%")
print("Matriz de confusão:")
print(confusion_matrix)