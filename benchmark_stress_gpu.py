import numpy as np
from utilidades import gerar_condicoes_iniciais
from simulacao_gpu import simular_n_corpos_gpu
import pycuda.driver as cuda
import pycuda.autoinit
#
# benchmark para testar os desempenhos da GPU em N muito grandes, onde o CPU já não é capaz de acompanhar.
# este script é independente do main.py e do benchmark_typst.py, focado exclusivamente em testar os limites de desempenho da GPU com N massivos.
# o script foi desenvolvido antes de otimizacoes finais, (dividir carga em multiplas GPUs)

def stress_test_gpu():
    # Valores SUPER massivos de N adaptados para GPUs de classe HPC (ex: A100 80GB)
    # potências de 2 alinham perfeitamente com os blocos de 256 threads
    lista_N = [10, 16384, 32768, 65536, 131072, 262144, 524288] # 
    
    # Usamos menos passos porque para 262144 partículas, O(N^2) significa 
    # mais de 68.7 mil milhões de interações *por passo*.
    PASSOS_TEMPO = 20 
    DELTA_T = 0.01
    TAMANHO_CAIXA = 100.0
    EPSILON = 1.0
    G = 1.0
    MASSA_MIN = 10.0
    MASSA_MAX = 50.0
    VELOCIDADE_MIN = -1.0
    VELOCIDADE_MAX = 1.0

    print("="*90)
    print(f" GPU STRESS TEST: Comparação de Arquiteturas de Memória CUDA ({PASSOS_TEMPO} Passos)")
    print("="*90)
    print(f" Dispositivo CUDA em uso : {cuda.Device(0).name()}")
    print("Neste teste o CPU é ignorado. O objetivo é saturar o poder de computação de uma GPU HPC.")
    print("-" * 90)
    print(f"{'N Partículas':<15} | {'Naive (s)':<15} | {'FastMath (s)':<15} | {'SharedMem (s)':<15} | {'Float4 (s)':<15}")
    print("-" * 90)

    for N in lista_N:
        np.random.seed(42)
        massas, posicoes, velocidades = gerar_condicoes_iniciais(
            N, TAMANHO_CAIXA, MASSA_MIN, MASSA_MAX, VELOCIDADE_MIN, VELOCIDADE_MAX
        )

        try:
            # 1. GPU Naive (Acesso cru à memória global)
            _, _, t_naive = simular_n_corpos_gpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='naive')
            
            # 2. GPU Naive + Fast Math (Apenas otimização matemática)
            _, _, t_fm = simular_n_corpos_gpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='naive_fast_math')
            
            # 3. GPU Shared Memory (Otimização de cache/Tiling)
            _, _, t_sm = simular_n_corpos_gpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='shared_mem')
            
            # 4. GPU Float4 (Otimização Máxima: Tiling + Coalesced Memory Access)
            _, _, t_f4 = simular_n_corpos_gpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='shared_mem_float4')

            # Imprimir linha de resultados
            print(f"{N:<15} | {t_naive:<15.4f} | {t_fm:<15.4f} | {t_sm:<15.4f} | {t_f4:<15.4f}")

        except Exception as e:
            print(f"{N:<15} | ERRO DE EXECUÇÃO: Provavelmente limite de memória da GPU atingido.")
            print(f"Detalhe do erro: {str(e)}")
            break # Para o loop se a GPU já não aguentar a quantidade de memória pedida

    print("="*90)

if __name__ == "__main__":
    stress_test_gpu()