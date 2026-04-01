import numpy as np
import matplotlib.pyplot as plt

def gerar_condicoes_iniciais(N, tamanho_caixa=100.0, massa_min=10.0, massa_max=50.0):
    """
    Gera as condições iniciais aleatórias para N partículas.
    Retorna arrays NumPy de massas, posições e velocidades (float32 para compatibilidade CUDA).
    """
    massas = np.random.uniform(massa_min, massa_max, N).astype(np.float32) # Massas random entre massa_min e massa_max
    posicoes = (np.random.rand(N, 3).astype(np.float32) * tamanho_caixa) - (tamanho_caixa / 2.0) # Posições random entre -tamanho_caixa/2 e +tamanho_caixa/2
    velocidades = np.random.uniform(-0.5, 0.5, (N, 3)).astype(np.float32) # Velocidades random entre -0.5 e 0.5 
    
    return massas, posicoes, velocidades

def desenhar_grafico_n_corpos(historico_posicoes, titulo="Simulação N-Corpos"):
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
        plt.plot(x, y, linewidth=0.5, alpha=0.7)
        # Ponto final da órbita
        plt.scatter(x[-1], y[-1], s=8)   
    plt.title(titulo)
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.axis('equal')
    plt.grid(True, alpha=0.15) # O N aqui é o número de partículas extraído do shape
    plt.savefig(f"{N}_corpos_orbitas.png", dpi=300, pad_inches=0)
    plt.show()