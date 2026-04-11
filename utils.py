import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from constants import MASSA_MIN, MASSA_MAX, VELOCIDADE_MIN, VELOCIDADE_MAX
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

def gerar_condicoes_iniciais_clusters(N, num_clusters=2, tamanho_caixa=50.0, dispersao_posicao=10.0, dispersao_velocidade=0.01):
    """
    Gera posições, velocidades e massas agrupadas em clusters (aglomerados).
    
    :param n_corpos: Número total de partículas na simulação.
    :param num_clusters: Quantidade de aglomerados para gerar.
    :param tamanho_caixa: O espaço dentro do qual os centros dos clusters serão espalhados.
    :param dispersao_posicao: Quão 'espalhado' o cluster é (desvio padrão da posição).
    :param dispersao_velocidade: Quão caóticas são as velocidades dentro do cluster.
    """
    posicoes = np.zeros((N, 3), dtype=np.float32)
    velocidades = np.zeros((N, 3), dtype=np.float32)
    massas = np.random.uniform(MASSA_MIN, MASSA_MAX, N).astype(np.float32)
    
    particulas_por_cluster = N // num_clusters
    
    for i in range(num_clusters):
        inicio = i * particulas_por_cluster
        # O último cluster recebe qualquer sobra de partículas caso a divisão não seja exata
        fim = N if i == num_clusters - 1 else (i + 1) * particulas_por_cluster
        n_particulas_aqui = fim - inicio
        
        # 1. Definir o centro deste cluster e a sua velocidade "média" (direção de movimento do cluster inteiro)
        centro_cluster = np.random.uniform(-tamanho_caixa / 2, tamanho_caixa / 2, 3)
        
        # Mantém a velocidade do cluster dentro dos limites do constants.py
        vel_media_cluster = np.random.uniform(VELOCIDADE_MIN, VELOCIDADE_MAX, 3) * 0.5 
        
        # 2. Gerar as partículas ao redor do centro usando uma distribuição Normal (Gaussiana)
        posicoes[inicio:fim] = np.random.normal(loc=centro_cluster, scale=dispersao_posicao, size=(n_particulas_aqui, 3))
        
        # 3. Gerar as velocidades das partículas ao redor da velocidade média do cluster
        # Usamos uma dispersão pequena para que o cluster permaneça junto nos primeiros momentos
        velocidades[inicio:fim] = np.random.normal(loc=vel_media_cluster, scale=dispersao_velocidade, size=(n_particulas_aqui, 3))
        
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

def gerar_grafico_tempo(nome_ficheiro, lista_n, **series_dados):
    """
    Gera um gráfico de performance genérico a partir de um dicionário de séries de dados.

    Args:
        nome_ficheiro (str): Nome do ficheiro para guardar o gráfico.
        lista_n (list): Lista de valores para o eixo X (e.g., número de partículas).
        **series_dados: Dicionário onde a chave é o label da série e o valor é uma
                        lista de tempos de execução. Ex: {'CPU': tempos_cpu, 'GPU': tempos_gpu}
    """
    plt.figure(figsize=(10, 6))

    log_y = series_dados.pop('log_y', True)
    log_x = series_dados.pop('log_x', True)
    

    for label, tempos in series_dados.items():
        plt.plot(lista_n, tempos, marker='o', label=label)

    if log_y:
        plt.yscale('log', base=10)
    if log_x:
        plt.xscale('log', base=2)
    plt.xlabel('Número de Partículas (N)')
    plt.ylabel('Tempo de Execução (s)')
    plt.title('Tempo de Execução vs Número de Partículas')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(nome_ficheiro, dpi=300)
    print(f"Gráfico gerado e guardado como '{nome_ficheiro}'")

def gerar_tabela_typst_single_gpu(nome_ficheiro, resultados):
    """
    Gera uma tabela formatada para Typst com os resultados do benchmark single-GPU.

    Args:
        nome_ficheiro (str): O nome do ficheiro de saída.
        resultados (list[dict]): Uma lista de dicionários, onde cada dicionário
                                 contém os resultados para um valor de N.
    """
    with open(nome_ficheiro, "w", encoding="utf-8") as f:
        f.write("  #align(center)[\n")
        f.write("  #table(\n")
        f.write("    columns: (auto, auto, auto, auto, auto, auto, auto, auto),\n")
        f.write("    inset: 5pt,\n")
        f.write("    align: horizon,\n")
        f.write(r"    [*N-Corpos*], [*t_CPU (s)*], [*Speedup\ naive*], [*Speedup\ fast math*], [*Speedup\ Mem. Part.*], [*Speedup\ vetores `float4`*],[*Desvio\ Máx. *], [*Erro\ Energia*]," + "\n")

        for res in resultados:
            f.write(
                f"    [{res['N']}], [{res['t_cpu']:.4f}], "
                f"[{res['speedup_naive']:.2f}x], [{res['speedup_fm']:.2f}x], "
                f"[{res['speedup_sm']:.2f}x], [{res['speedup_f4']:.2f}x], "
                f"[{res['desvio']:.6f}], [{res['erro_energia']:.5f}%],\n"
            )

        f.write("  )\n")
        f.write("  ]\n")
    print(f"\nSUCESSO! Tabela gerada e guardada no ficheiro '{nome_ficheiro}'")

def gerar_tabela_typst_multi_gpu(nome_ficheiro, resultados):
    """
    Gera uma tabela formatada para Typst com os resultados do benchmark multi-GPU.

    Args:
        nome_ficheiro (str): O nome do ficheiro de saída.
        resultados (list[dict]): Uma lista de dicionários, onde cada dicionário
                                 contém os resultados para um valor de N.
    """
    with open(nome_ficheiro, "w", encoding="utf-8") as f:
        f.write("  #align(center)[\n")
        f.write("  #table(\n")
        f.write("    columns: (auto, auto, auto, auto, auto, auto, auto, auto),\n")
        f.write("    inset: 5pt,\n")
        f.write("    align: horizon,\n")
        f.write(r"    [*N*], [*1-GPU\ `float4` (s)*], [*Multi PCIe (s)*],  [*Multi NVLink (s)*], [*Speedup PCIe*], [*Speedup NVLink*], [*Desvio\ Máx.*], [*Erro\ Energia*]," + "\n")

        for res in resultados:
            f.write(
                f"    [{res['N']}], [{res['t_1gpu']:.4f}], [{res['t_multigpu']:.4f}], "
                f"[{res['t_nv4']:.4f}], [*{res['speedup_multi']:.2f}x*], "
                f"[*{res['speedup_nv4']:.2f}x*], [{res['desvio']:.6f}], "
                f"[{res['erro_energia']:.5f}%],\n"
            )

        f.write("  )\n")
        f.write("  ]\n")
    print(f"\nSUCESSO! Tabela gerada e guardada no ficheiro '{nome_ficheiro}'")