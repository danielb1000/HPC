import time
import numpy as np
from utilidades import gerar_condicoes_iniciais
from simulacao_cpu import simular_n_corpos_cpu
from simulacao_gpu import simular_n_corpos_gpu, validar_energia_gpu
import pycuda.autoinit
#
# benchmark para gerar a tabela de resultados para o Typst antes de otimizacoes finais ()
# Este script é independente do main.py e do benchmark_stress_gpu.py.
# 

def gerar_tabela_typst():
    # Os valores de N que queremos testar
    lista_N = [2, 8, 32, 256, 512, 1024, 2048, 8192, 16384, 32768] # Potências de 2 para alinhamento perfeito com blocos de 512 threads
    
    # Constantes da simulação
    TAMANHO_CAIXA = False # O tamanho vai ser dinâmico para cada N_PARTICULAS
    PASSOS_TEMPO = 200
    DELTA_T = 0.01
    EPSILON = 1.0
    G = 1.0
    MASSA_MIN = 10.0
    MASSA_MAX = 50.0
    VELOCIDADE_MIN = -1.0
    VELOCIDADE_MAX = 1.0

    print("A iniciar o benchmark. Isto pode demorar alguns minutos...")
    
    with open("tabela_typst.txt", "w", encoding="utf-8") as f:
        f.write("  #align(center)[\n")
        f.write("  #table(\n")
        f.write("    columns: (auto, auto, auto, auto, auto, auto, auto, auto),\n")
        f.write("    inset: 5pt,\n")
        f.write("    align: horizon,\n")
        f.write("    [*N-Corpos*], [*t_CPU (s)*], [*Speedup\\ naive*], [*Speedup\\ fast math*], [*Speedup\\ Mem. Part.*], [*Speedup\\ vetores `float4`*],[*Desvio\\ Máx. *], [*Erro\\ Energia*],\n")

        for N_PARTICULAS in lista_N:
            # Calcular tamanho da caixa dinamicamente para manter densidade constante (~0.002)
            # Evita que Ns pequenos fiquem demasiado dispersos e Ns grandes explodam em NaNs.
            TAMANHO_CAIXA = np.cbrt(N_PARTICULAS / 0.002)

            np.random.seed(42) # Seed fixa para cada N ser consistente
            massas, posicoes, velocidades = gerar_condicoes_iniciais(
                N_PARTICULAS, TAMANHO_CAIXA, MASSA_MIN, MASSA_MAX, VELOCIDADE_MIN, VELOCIDADE_MAX
            )
            
            energia_inicial = validar_energia_gpu(posicoes, velocidades, massas, G, EPSILON)

            # 1. CPU
            pos_cpu = posicoes.copy()
            inicio_cpu = time.perf_counter()
            simular_n_corpos_cpu(pos_cpu, velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, guardar_historico=False)
            t_cpu = time.perf_counter() - inicio_cpu

            # 2. GPU Naive
            _, _, t_gpu_naive = simular_n_corpos_gpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='naive')

            # 3. GPU Naive Fast Math
            _, _, t_gpu_fm = simular_n_corpos_gpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='naive_fast_math')

            # 4. GPU Shared Memory
            _, _, t_gpu_sm = simular_n_corpos_gpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='shared_mem')

            # 5. GPU Float4
            pos_gpu_f4, vel_gpu_f4, t_gpu_f4 = simular_n_corpos_gpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='shared_mem_float4')

            desvio = np.max(np.abs(pos_cpu - pos_gpu_f4))
            
            energia_final = validar_energia_gpu(pos_gpu_f4, vel_gpu_f4, massas, G, EPSILON)
            erro_energia = abs((energia_final - energia_inicial) / energia_inicial) * 100

            speedup_naive = t_cpu / t_gpu_naive
            speedup_fm = t_cpu / t_gpu_fm
            speedup_sm = t_cpu / t_gpu_sm
            speedup_f4 = t_cpu / t_gpu_f4

            f.write(f"    [{N_PARTICULAS}], [{t_cpu:.4f}], [{speedup_naive:.2f}x], [{speedup_fm:.2f}x], [{speedup_sm:.2f}x], [{speedup_f4:.2f}x], [{desvio:.6f}], [{erro_energia:.5f}%],\n")
            print(f" -> Concluído benchmark para N = {N_PARTICULAS}")

        f.write("  )\n")
        f.write("  ]\n")
        
    print("\nSUCESSO! Tabela gerada e guardada no ficheiro 'tabela_typst.txt'")

if __name__ == "__main__":
    gerar_tabela_typst()