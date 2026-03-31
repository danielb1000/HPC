import numpy as np
import matplotlib.pyplot as plt

def gerar_condicoes_iniciais(N, tamanho_caixa=100.0, massa_min=10.0, massa_max=50.0):
    """
    Gera as condições iniciais aleatórias para N partículas.
    Retorna arrays NumPy de massas, posições e velocidades (float32 para compatibilidade CUDA).
    """
    # Massas aleatórias no intervalo [10.0, 50.0]
    massas = np.random.uniform(massa_min, massa_max, N).astype(np.float32)
    
    # Posições 3D num volume de tamanho_caixa^3 centrado na origem
    posicoes = (np.random.rand(N, 3).astype(np.float32) * tamanho_caixa) - (tamanho_caixa / 2.0)
    
    # Velocidades iniciais com ruído aleatório entre -0.5 e 0.5
    velocidades = np.random.uniform(-0.5, 0.5, (N, 3)).astype(np.float32)
    
    return massas, posicoes, velocidades

def desenhar_grafico_n_corpos(historico_posicoes, titulo="Simulação N-Corpos", N = None):
    """
    Recebe um histórico de posições com shape (P, N, 3)
    e desenha o rasto em 2D projetando os eixos X e Y.
    """
    hist_np = np.array(historico_posicoes)
    P, N, _ = hist_np.shape
    
    fig = plt.figure(figsize=(8, 8)) # Obter o objeto da figura
    plt.style.use('dark_background') # Aplicar o estilo ao contexto atual da figura 'fig'
    fig.set_facecolor('black') # Definir explicitamente o fundo da figura para preto
    
    # Iterar sobre cada partícula e desenhar a sua trajetória
    for i in range(N):
        x = hist_np[:, i, 0]
        y = hist_np[:, i, 1]
        plt.plot(x, y, linewidth=0.5, alpha=0.7)
        # Ponto final da órbita
        plt.scatter(x[-1], y[-1], s=8)
        
    plt.title(titulo)
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.axis('equal')
    plt.grid(True, alpha=0.15) # Manter a grade
    
    if N is not None:
        plt.savefig(f"{N}_corpos_orbitas.png", dpi=300, pad_inches=0)
    plt.show()