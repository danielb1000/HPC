import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def calcular_tamanho_caixa_dinamico(N_particulas, densidade_alvo=0.002):
    """
    Calcula o tamanho da caixa de simulação dinamicamente para manter uma densidade constante.
    Evita que Ns pequenos fiquem demasiado dispersos e Ns grandes explodam em NaNs.
    """
    return np.cbrt(N_particulas / densidade_alvo)


def gerar_condicoes_iniciais(N, tamanho_caixa=100.0, massa_min=10.0, massa_max=50.0,velocidade_min=-0.5 ,velocidade_max=0.5):
    """
    Gera as condições iniciais aleatórias para N partículas.
    Retorna arrays NumPy de massas, posições e velocidades (float32 para compatibilidade CUDA).
    """
    massas = np.random.uniform(massa_min, massa_max, N).astype(np.float32) # Massas random entre massa_min e massa_max
    posicoes = (np.random.rand(N, 3).astype(np.float32) * tamanho_caixa) - (tamanho_caixa / 2.0) # Posições random entre -tamanho_caixa/2 e +tamanho_caixa/2
    velocidades = np.random.uniform(velocidade_min, velocidade_max, (N, 3)).astype(np.float32) # Velocidades random entre velocidade_min e velocidade_max
    
    return massas, posicoes, velocidades

def desenhar_grafico_n_corpos(historico_posicoes, massas, titulo="Simulação N-Corpos"):
    """
    Recebe um histórico de posições com shape (P, N, 3)
    e desenha o rasto em 2D projetando os eixos X e Y.
    """
    hist_np = np.asarray(historico_posicoes) # Usar asarray é mais eficiente se já for um np.array
    P, N, _ = hist_np.shape # Extrair o número de passos (P) e partículas (N) do histórico
    
    # Configurações e criação do gráfico
    fig = plt.figure(figsize=(8, 8))
    plt.style.use('dark_background') 
    fig.set_facecolor('black') 
    for i in range(N):
        x = hist_np[:, i, 0] # Coordenada X da partícula i ao longo do tempo
        y = hist_np[:, i, 1] 
        plt.plot(x, y, linewidth=1, alpha=0.6)
        # Ponto final da órbita, com tamanho proporcional à massa
        plt.scatter(x[-1], y[-1], s=massas[i] * 1, linewidths=0.8, alpha=0.9)   
    plt.title(titulo)
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.axis('equal')
    plt.grid(True, alpha=0.15) # O N aqui é o número de partículas extraído do shape
    print("saving figure...")
    plt.savefig(f"{N}_corpos_orbitas.png", dpi=300, pad_inches=0)
    print(f"figure saved as {N}_corpos_orbitas.png")
    # plt.show()

def gerar_grafico_performance(nome_ficheiro, lista_n, **series_dados):
    """
    Gera um gráfico de performance genérico a partir de um dicionário de séries de dados.

    Args:
        nome_ficheiro (str): Nome do ficheiro para guardar o gráfico.
        lista_n (list): Lista de valores para o eixo X (e.g., número de partículas).
        **series_dados: Dicionário onde a chave é o label da série e o valor é uma
                        lista de tempos de execução. Ex: {'CPU': tempos_cpu, 'GPU': tempos_gpu}
    """
    plt.figure(figsize=(10, 6))
    for label, tempos in series_dados.items():
        plt.plot(lista_n, tempos, marker='o', label=label)

    plt.yscale('log', base=10)
    plt.xlabel('Número de Partículas (N)')
    plt.ylabel('Tempo de Execução (s)')
    plt.title('Tempo de Execução vs Número de Partículas')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(nome_ficheiro, dpi=300)
    print(f"Gráfico gerado e guardado como '{nome_ficheiro}'")