import time
import numpy as np
import multiprocessing as mp
import pycuda.driver as cuda
from utilidades import (gerar_condicoes_iniciais, calcular_tamanho_caixa_dinamico, 
                        gerar_tabela_typst_multi_gpu,
                        gerar_grafico_tempo)
from simulacao_gpu import simular_n_corpos_gpu, validar_energia_gpu
from simulacao_multigpu import simular_n_corpos_multigpu
from simulacao_nv4 import simular_n_corpos_nv4

#
# Benchmark exclusivo para testar a escalabilidade Multi-GPU em escalas de HPC
# e exportar os resultados formatados para o relatório Typst.
# 

def executar_benchmark_multi_gpu():
    # Valores massivos para demonstrar o ponto onde o overhead de comunicação é 
    # ultrapassado pelo poder bruto de paralelismo das 4 GPUs.
    lista_N = [32, 128, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144] # Potências de 2 para alinhamento perfeito com blocos 
    from constants import PASSOS_TEMPO, DELTA_T, EPSILON, G, MASSA_MIN, MASSA_MAX, VELOCIDADE_MIN, VELOCIDADE_MAX, DENSIDADE_ALVO

    cuda.init()
    num_gpus = cuda.Device.count()
    
    print(f"A iniciar o benchmark Typst para Multi-GPU ({num_gpus} GPUs detetadas).")
    print("Isto pode demorar alguns minutos...\n")
    
    # Estrutura para definir as simulações a serem executadas em ciclo
    simulations_to_run = [
        ('1gpu', simular_n_corpos_gpu, {'method': 'shared_mem_float4'}),
        ('multigpu', simular_n_corpos_multigpu, {'num_gpus_ativas': num_gpus}),
        ('nv4', simular_n_corpos_nv4, {'num_gpus_ativas': num_gpus})
    ]

    # Contexto principal isolado para a simulação 1-GPU e cálculos de energia
    ctx = cuda.Device(0).make_context()

    # Lista para armazenar os resultados de forma estruturada para a tabela
    resultados_tabela = []
    
    # Dicionário para armazenar listas de tempos para o gráfico
    tempos_para_grafico = {'1gpu': [], 'multigpu': [], 'nv4': []}
    
    try:
        for N_PARTICULAS in lista_N:
            # Calcular tamanho da caixa dinamicamente
            TAMANHO_CAIXA = calcular_tamanho_caixa_dinamico(N_PARTICULAS, DENSIDADE_ALVO)

            np.random.seed(42) # Seed fixa para reprodutibilidade
            massas, posicoes, velocidades = gerar_condicoes_iniciais(
                N_PARTICULAS, TAMANHO_CAIXA, MASSA_MIN, MASSA_MAX, VELOCIDADE_MIN, VELOCIDADE_MAX
            )
            
            energia_inicial = validar_energia_gpu(posicoes, velocidades, massas, G, EPSILON)

            # Dicionários para armazenar os resultados desta iteração de N
            tempos_n = {}
            posicoes_finais_n = {}
            velocidades_finais_n = {}

            # Executar todas as simulações definidas em `simulations_to_run`
            for key, func, kwargs in simulations_to_run:
                pos_final, vel_final, tempo = func(
                    posicoes.copy(), velocidades.copy(), massas, 
                    PASSOS_TEMPO, DELTA_T, G, EPSILON, **kwargs
                )
                tempos_n[key] = tempo
                posicoes_finais_n[key] = pos_final
                velocidades_finais_n[key] = vel_final

            # Validações Físicas
            desvio_multigpu = np.max(np.abs(posicoes_finais_n['1gpu'] - posicoes_finais_n['multigpu']))
            desvio_nv4 = np.max(np.abs(posicoes_finais_n['1gpu'] - posicoes_finais_n['nv4']))
            desvio = max(desvio_multigpu, desvio_nv4)

            energia_final = validar_energia_gpu(posicoes_finais_n['nv4'], velocidades_finais_n['nv4'], massas, G, EPSILON)
            erro_energia = abs((energia_final - energia_inicial) / energia_inicial) * 100
            speedup_multi = tempos_n['1gpu'] / tempos_n['multigpu']
            speedup_nv4 = tempos_n['1gpu'] / tempos_n['nv4']

            # Guardar resultados para a tabela
            resultados_tabela.append({
                'N': N_PARTICULAS,
                't_1gpu': tempos_n['1gpu'],
                't_multigpu': tempos_n['multigpu'],
                't_nv4': tempos_n['nv4'],
                'speedup_multi': speedup_multi,
                'speedup_nv4': speedup_nv4,
                'desvio': desvio,
                'erro_energia': erro_energia
            })
            
            # Adicionar tempos às listas para o gráfico
            tempos_para_grafico['1gpu'].append(tempos_n['1gpu'])
            tempos_para_grafico['multigpu'].append(tempos_n['multigpu'])
            tempos_para_grafico['nv4'].append(tempos_n['nv4'])
            
            print(f" -> Concluído benchmark para N = {N_PARTICULAS} | Speedup Multi-GPU: {speedup_multi:.2f}x | Speedup NVLink: {speedup_nv4:.2f}x")

        # Gerar a tabela Typst usando a função de utilidade
        gerar_tabela_typst_multi_gpu("benchmark_multigpu_performance.txt", resultados_tabela)
        
        # Gerar o gráfico de performance
        series_para_plotar = {
            '1-GPU (Float4)': tempos_para_grafico['1gpu'],
            'Multi-GPU (PCIe)': tempos_para_grafico['multigpu'],
            'Multi-GPU (NVLink)': tempos_para_grafico['nv4']
        }
        gerar_grafico_tempo('benchmark_multigpu_performance.png', lista_N, log_y=False, log_x=False, **series_para_plotar)
        
    finally:
        ctx.pop()

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    executar_benchmark_multi_gpu()