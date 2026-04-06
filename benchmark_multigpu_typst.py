import time
import numpy as np
import multiprocessing as mp
import pycuda.driver as cuda
from utilidades import gerar_condicoes_iniciais
from simulacao_gpu import simular_n_corpos_gpu, validar_energia_gpu
from simulacao_multigpu import simular_n_corpos_multigpu

#
# Benchmark exclusivo para testar a escalabilidade Multi-GPU em escalas de HPC
# e exportar os resultados formatados para o relatório Typst.
# 

def gerar_tabela_multigpu_typst():
    # Valores massivos para demonstrar o ponto onde o overhead de comunicação é 
    # ultrapassado pelo poder bruto de paralelismo das 4 GPUs.
    lista_N = [16384, 32768, 65536, 131072, 262144, 524288]
    
    PASSOS_TEMPO = 20
    DELTA_T = 0.01
    TAMANHO_CAIXA = 2000.0 # Caixa larga para evitar explosões físicas por super-densidade
    EPSILON = 1.0
    G = 1.0
    MASSA_MIN = 10.0
    MASSA_MAX = 50.0
    VELOCIDADE_MIN = -1.0
    VELOCIDADE_MAX = 1.0

    cuda.init()
    num_gpus = cuda.Device.count()
    
    print(f"A iniciar o benchmark Typst para Multi-GPU ({num_gpus} GPUs detetadas).")
    print("Isto pode demorar alguns minutos...\n")
    
    # Contexto principal isolado para a simulação 1-GPU e cálculos de energia
    ctx = cuda.Device(0).make_context()
    
    try:
        with open("tabela_multigpu_typst.txt", "w", encoding="utf-8") as f:
            f.write("  #align(center)[\n")
            f.write("  #table(\n")
            f.write("    columns: (auto, auto, auto, auto, auto, auto),\n")
            f.write("    inset: 5pt,\n")
            f.write("    align: horizon,\n")
            f.write("    [*N-Corpos*], [*1 GPU Float4 (s)*], [*Multi-GPU (s)*], [*Speedup*], [*Desvio Máx.*], [*Erro Energia*],\n")

            for N in lista_N:
                np.random.seed(42)
                massas, posicoes, velocidades = gerar_condicoes_iniciais(
                    N, TAMANHO_CAIXA, MASSA_MIN, MASSA_MAX, VELOCIDADE_MIN, VELOCIDADE_MAX
                )
                
                energia_inicial = validar_energia_gpu(posicoes, velocidades, massas, G, EPSILON)

                # 1. Baseline HPC: 1 GPU Otimização Máxima (Float4)
                pos_1gpu, vel_1gpu, t_1gpu = simular_n_corpos_gpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='shared_mem_float4')

                # 2. Distribuído: 4 GPUs
                pos_multigpu, vel_multigpu, t_multigpu = simular_n_corpos_multigpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, num_gpus_ativas=num_gpus)

                # Validações Físicas
                desvio = np.max(np.abs(pos_1gpu - pos_multigpu))
                energia_final = validar_energia_gpu(pos_multigpu, vel_multigpu, massas, G, EPSILON)
                erro_energia = abs((energia_final - energia_inicial) / energia_inicial) * 100
                speedup = t_1gpu / t_multigpu

                f.write(f"    [{N}], [{t_1gpu:.4f}], [{t_multigpu:.4f}], [*{speedup:.2f}x*], [{desvio:.6f}], [{erro_energia:.5f}%],\n")
                print(f" -> Concluído benchmark para N = {N} | Speedup Multi-GPU: {speedup:.2f}x")

            f.write("  )\n")
            f.write("  ]\n")
            
        print("\nSUCESSO! Tabela gerada e guardada no ficheiro 'tabela_multigpu_typst.txt'")
    finally:
        ctx.pop()

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    gerar_tabela_multigpu_typst()