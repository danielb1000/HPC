import numpy as np
import time

def calcular_aceleracoes_numpy(posicoes, massas, G, eps):
    """
    Calcula a aceleração de N corpos usando vetorização e broadcasting do NumPy.
    Resolve a complexidade O(N^2) sem iterações 'for' explícitas.
    """
    # 1. Expandir dimensões: posicoes (N, 3) -> pos_i (N, 1, 3) e pos_j (1, N, 3)
    pos_i = posicoes[:, np.newaxis, :]
    pos_j = posicoes[np.newaxis, :, :]
    
    # 2. Matriz de distâncias relativas: shape (N, N, 3)
    dx = pos_j - pos_i 
    
    # 3. Quadrado das distâncias reais (soma das componentes x,y,z): shape (N, N)
    dist_sq = np.sum(dx**2, axis=-1)

    # 4. Denominador da gravidade suavizado pelo parâmetro epsilon
    np.fill_diagonal(dist_sq, 1.0)  # Evitar divisão por zero na diagonal (auto-interação)
    # Forçar o cálculo em float32 para ser uma comparação justa com a GPU.
    # O NumPy por defeito promove para float64 se um dos operandos for um float de Python.
    denominador = (dist_sq + eps**2)**np.float32(1.5)

    # 5. Formatar a massa para broadcasting: shape (1, N, 1)
    m_j = massas[np.newaxis, :, np.newaxis]

    # 6. Cálculo do tensor de forças e redução somando ao longo do eixo das colunas
    acel_matriz = G * m_j * dx / denominador[:, :, np.newaxis]
    aceleracoes_finais = np.sum(acel_matriz, axis=1)
    
    return aceleracoes_finais

def simular_n_corpos_cpu(pos, vel, massas, passos, dt, G, eps, callback_progresso=None, guardar_historico=False):
    """
    Integra as equações de movimento usando o método Velocity Verlet.
    Se `guardar_historico` for True, devolve o histórico de posições para visualização.
    Por defeito, `guardar_historico` é False para permitir benchmarks de performance justos,
    onde a função apenas executa a simulação e modifica os arrays `pos` e `vel` in-place.
    """
    # Forçar todos os escalares para float32 para garantir que a simulação CPU
    # usa a mesma precisão que a GPU, evitando discrepâncias de precisão que podem ocorrer se o NumPy promover para float64.
    dt = np.float32(dt)
    G = np.float32(G)
    eps = np.float32(eps)

    N = pos.shape[0]
    historico_posicoes = None
    if guardar_historico:
        # MODO LENTO: Aloca um grande array para guardar o histórico completo.
        # Necessário para visualização, mas muito custoso para benchmarks.
        historico_posicoes = np.zeros((passos, N, 3), dtype=np.float32)
    
    # Aceleração inicial (t=0)
    acel = calcular_aceleracoes_numpy(pos, massas, G, eps)
    
    # Calcular o intervalo para imprimir o progresso (ex: a cada 20%)
    passo_progresso = max(1, passos // 20) if callback_progresso else passos + 1
    
    for passo in range(passos):
        # OTIMIZAÇÃO CRÍTICA PARA BENCHMARK:
        # Esta cópia de dados é a operação mais lenta depois do cálculo de aceleração.
        # Ao usar `guardar_historico=False`, saltamos este passo e obtemos um tempo de CPU justo.
        if guardar_historico:
            historico_posicoes[passo] = pos.copy()
        
        # --- VELOCITY VERLET ---
        # 1. Posição atualizada com termo da aceleração
        pos += vel * dt + np.float32(0.5) * acel * (dt**2)
        
        # 2. Avaliação de forças na nova posição
        nova_acel = calcular_aceleracoes_numpy(pos, massas, G, eps)
        
        # 3. Correção da velocidade (half-step implícito na média)
        vel += np.float32(0.5) * (acel + nova_acel) * dt
        
        acel = nova_acel
        
        # --- FEEDBACK DA SIMULAÇÃO ---
        # Apenas tenta entrar neste bloco se um callback foi realmente fornecido (não é None).
        if callback_progresso and ((passo + 1) % passo_progresso == 0 or (passo + 1) == passos):
            callback_progresso(passo + 1, passos) # Esta chamada agora é segura.
            
    return historico_posicoes