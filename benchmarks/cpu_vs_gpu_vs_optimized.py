import time
import numpy as np
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
current_dir = Path(__file__).parent
from utils import (gerar_condicoes_iniciais, gerar_grafico_tempo, 
                        calcular_tamanho_caixa_dinamico, gerar_tabela_typst_single_gpu)
from engines.cpu import simular_n_corpos_cpu
from engines.gpu import simular_n_corpos_gpu, validar_energia_gpu
import pycuda.autoinit

#
# benchmark para gerar a tabela de resultados para o Typst antes de otimizacoes finais ()
# Este script é independente do main.py e do benchmark_stress_gpu.py.
# 

def executar_benchmark_single_gpu():
    # Os valores de N que queremos testar
    lista_N = [2, 4, 8, 16, 32, 64, 128, 256, 512] # Potências de 2 para alinhamento perfeito com blocos de 512 threads
    
    from constants import PASSOS_TEMPO, DELTA_T, EPSILON, G, MASSA_MIN, MASSA_MAX, VELOCIDADE_MIN, VELOCIDADE_MAX, DENSIDADE_ALVO

    print("A iniciar o benchmark single-GPU. Isto pode demorar alguns minutos...")

    # Definir True para métodos a testar
    metodos_gpu = {
        'naive': True,              # PyCUDA - Baseline (Naive)
        'naive_fast_math': True,   # PyCUDA - Naive + Fast Math
        'shared_mem': True,         # PyCUDA - Shared Memory + Fast Math
        'shared_mem_float4': True,  # PyCUDA - Shared Mem + Fast Math + Float4 Vector
    }

    # Dicionário para armazenar listas de tempos para o gráfico
    tempos_para_grafico = {metodo: [] for metodo, executar in metodos_gpu.items() if executar}
    tempos_para_grafico['cpu'] = []

    # Lista para armazenar os resultados de forma estruturada para a tabela
    resultados_tabela = []

    for N_PARTICULAS in lista_N:
        # Calcular tamanho da caixa dinamicamente
        TAMANHO_CAIXA = calcular_tamanho_caixa_dinamico(N_PARTICULAS, DENSIDADE_ALVO)

        np.random.seed(42) # Seed fixa para cada N ser consistente para reprodutibilidade
        massas, posicoes, velocidades = gerar_condicoes_iniciais(
            N_PARTICULAS, TAMANHO_CAIXA, MASSA_MIN, MASSA_MAX, VELOCIDADE_MIN, VELOCIDADE_MAX
        )
        
        energia_inicial = validar_energia_gpu(posicoes, velocidades, massas, G, EPSILON)

        # Dicionário para armazenar os tempos desta iteração de N
        tempos_n = {}

        # 1. CPU
        pos_cpu = posicoes.copy()
        inicio_cpu = time.perf_counter()
        simular_n_corpos_cpu(pos_cpu, velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, guardar_historico=False)
        tempos_n['cpu'] = time.perf_counter() - inicio_cpu

        # 2. GPU (todas as versões)
        pos_gpu_final, vel_gpu_final = None, None
        for metodo, executar in metodos_gpu.items():
            if executar:
                pos_gpu_final, vel_gpu_final, t_gpu = simular_n_corpos_gpu(
                    posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method=metodo
                )
                tempos_n[metodo] = t_gpu

        # Adicionar tempos às listas para o gráfico
        tempos_para_grafico['cpu'].append(tempos_n['cpu'])
        for metodo, executar in metodos_gpu.items():
            if executar:
                tempos_para_grafico[metodo].append(tempos_n[metodo])

        # Validações usam a última simulação GPU (a mais otimizada)
        desvio = np.max(np.abs(pos_cpu - pos_gpu_final))
        
        energia_final = validar_energia_gpu(pos_gpu_final, vel_gpu_final, massas, G, EPSILON)
        erro_energia = abs((energia_final - energia_inicial) / energia_inicial) * 100

        # Guardar resultados para a tabela
        resultados_tabela.append({
            'N': N_PARTICULAS,
            't_cpu': tempos_n['cpu'],
            'speedup_naive': tempos_n['cpu'] / tempos_n['naive'] if 'naive' in tempos_n else 0.0,
            'speedup_fm': tempos_n['cpu'] / tempos_n['naive_fast_math'] if 'naive_fast_math' in tempos_n else 0.0,
            'speedup_sm': tempos_n['cpu'] / tempos_n['shared_mem'] if 'shared_mem' in tempos_n else 0.0,
            'speedup_f4': tempos_n['cpu'] / tempos_n['shared_mem_float4'] if 'shared_mem_float4' in tempos_n else 0.0,
            'desvio': desvio,
            'erro_energia': erro_energia
        })
        print(f" -> Concluído benchmark para N = {N_PARTICULAS}")

    # Gerar a tabela Typst usando a função de utilidade
    
    gerar_tabela_typst_single_gpu(str(current_dir / "benchmark_CPU_vs_GPU_vs_GPU_Optimized.txt"), resultados_tabela)


    # Gerar o gráfico de performance
    series_para_plotar = {'CPU': tempos_para_grafico['cpu']}
    if 'naive' in tempos_para_grafico:
        series_para_plotar['GPU Naive'] = tempos_para_grafico['naive']
    if 'shared_mem_float4' in tempos_para_grafico:
        series_para_plotar['GPU Optimized'] = tempos_para_grafico['shared_mem_float4']
        
    gerar_grafico_tempo(str(current_dir /'benchmark_CPU_vs_GPU_vs_GPU_Optimized.png'), lista_N, log_y=True, log_x=False, **series_para_plotar)


if __name__ == "__main__":
    executar_benchmark_single_gpu()