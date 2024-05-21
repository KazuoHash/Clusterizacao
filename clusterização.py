import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# Função para calcular a distância total de uma rota
def calcular_distancia_total(rota, matriz_distancias):
    distancia_total = 0
    for i in range(len(rota) - 1):
        origem = rota[i]
        destino = rota[i + 1]
        distancia_total += matriz_distancias[origem, destino]
    return distancia_total

# Função para encontrar a melhor rota usando o problema do caixeiro viajante
def encontrar_melhor_rota(pontos):
    num_pontos = len(pontos)
    matriz_distancias = distance_matrix(pontos, pontos)

    # Inicializa o modelo de atribuição linear
    linha_indice, coluna_indice = linear_sum_assignment(matriz_distancias)

    # Encontra a melhor rota usando o problema do caixeiro viajante
    melhor_rota = []
    for origem, destino in zip(linha_indice, coluna_indice):
        melhor_rota.append(origem)
        melhor_rota.append(destino)

    melhor_distancia = calcular_distancia_total(melhor_rota, matriz_distancias)
    return melhor_rota, melhor_distancia

# Gerar pontos de exemplo
np.random.seed(0)
pontos = np.random.rand(100, 2)

# Executar o algoritmo K-means
kmeans = KMeans(n_clusters=5)
kmeans.fit(pontos)

# Obter os centroides de cada cluster
centroides = kmeans.cluster_centers_

# Plotar os pontos e os centroides
plt.scatter(pontos[:, 0], pontos[:, 1], c=kmeans.labels_)
plt.scatter(centroides[:, 0], centroides[:, 1], marker='x', c='red')

# Interligar todos os pontos dentro de cada cluster
for i in range(kmeans.n_clusters):
    indices_cluster = np.where(kmeans.labels_ == i)[0]
    pontos_cluster = pontos[indices_cluster]
    melhor_rota_cluster, _ = encontrar_melhor_rota(pontos_cluster)

    # Plotar a rota dentro do cluster
    plt.plot(pontos_cluster[melhor_rota_cluster][:, 0], pontos_cluster[melhor_rota_cluster][:, 1])

# Encontrar a melhor rota entre os centroides
melhor_rota_centroides, _ = encontrar_melhor_rota(centroides)

# Plotar a rota entre os centroides
plt.plot(centroides[melhor_rota_centroides][:, 0], centroides[melhor_rota_centroides][:, 1], 'r--')

plt.show()