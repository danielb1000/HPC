import time
from utilidades import gerar_condicoes_iniciais, desenhar_grafico_n_corpos
from simulacao_cpu import simular_n_corpos_cpu
from simulacao_gpu import simular_n_corpos_gpu
import numpy as np

def main():
    N_PARTICULAS = 1000       # Aumentado para saturar a GPU (ex: 8192, 16384, 32768)
    DELTA_T = 0.01          # Tempo de passo
    PASSOS_TEMPO = 1000     # Reduzido para evitar que o CPU demore demasiadas horas
    TAMANHO_CAIXA = 100.0   # Espaço cúbico onde as partículas são inicialmente distribuídas  
    EPSILON = 1.0           # Softening parameter = 1.0 em vez de 0.1 para evitar divergências numéricas em simulações longas.
    G = 1.0                 # Constante gravitacional
    MASSA_MIN = 10.0        # Massa mínima das partículas
    MASSA_MAX = 50.0        # Massa máxima das partículas
    VELOCIDADE_MIN = -1.0   # Velocidade mínima das partículas
    VELOCIDADE_MAX = 1.0    # Velocidade máxima das partículas
    np.random.seed(42)      # Seed fixa para resultados consistentes

    print("="*75)
    print(f" BENCHMARK N-CORPOS: {N_PARTICULAS} Partículas, {PASSOS_TEMPO} Passos, Δt={DELTA_T}, Espaço = {TAMANHO_CAIXA}^3")
    print("="*75)
    
    # Geração dos tensores (NumPy) de massas, posições e velocidades iniciais
    massas, posicoes, velocidades = gerar_condicoes_iniciais(
        N_PARTICULAS, TAMANHO_CAIXA, MASSA_MIN, MASSA_MAX, VELOCIDADE_MIN,VELOCIDADE_MAX, 
        )

    # Criar cópias para CPU e GPU para garantir que ambos começam com as mesmas condições iniciais
    pos_para_cpu = posicoes.copy()
    vel_para_cpu = velocidades.copy()

    # -- 1. CPU SIMULAÇÃO --
    print("[1/4] A executar na CPU (NumPy)...")

    inicio_cpu = time.perf_counter()
    # Para um benchmark de performance justo, chamamos a função com `guardar_historico=False`.
    # O histórico para o gráfico será gerado depois, se necessário.
    simular_n_corpos_cpu(pos_para_cpu, vel_para_cpu, massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, guardar_historico=False)
    tempo_cpu = time.perf_counter() - inicio_cpu
    pos_final_cpu = pos_para_cpu  # O resultado final está no array que foi passado e modificado.
    
    # -- 2. GPU SIMULAÇÃO (BASELINE (NAIVE)) --
    print("[2/4] A executar na GPU (PyCUDA - Baseline (Naive))...")
    pos_final_gpu_naive, tempo_gpu_naive = simular_n_corpos_gpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='naive')

    # -- 3. GPU SIMULAÇÃO (OTIMIZADA) --
    print("[3/4] A executar na GPU (PyCUDA - Otimizado com Shared Memory)...")
    pos_final_gpu_opt, tempo_gpu_opt = simular_n_corpos_gpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='shared_mem')

    # -- 4. GPU SIMULAÇÃO (VECTOR TYPES) --
    print("[4/4] A executar na GPU (PyCUDA - Memória de Banda Larga float4)...")
    pos_final_gpu_vec, tempo_gpu_vec = simular_n_corpos_gpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='shared_mem_float4')

    # VALIDAÇÃO MATEMÁTICA E RESULTADOS
    # O desvio máximo entre as posições finais da CPU e GPU deve ser pequeno (dependendo do softening e do número de passos).
    desvio_cpu_vs_gpu_naive = np.max(np.abs(pos_final_cpu - pos_final_gpu_naive))
    desvio_cpu_vs_gpu_opt = np.max(np.abs(pos_final_cpu - pos_final_gpu_opt))
    desvio_cpu_vs_gpu_vec = np.max(np.abs(pos_final_cpu - pos_final_gpu_vec))

    print("\n" + "="*80)
    print(" RESULTADOS DO BENCHMARK para" ,N_PARTICULAS, "partículas,", PASSOS_TEMPO, "passos, Δt=", DELTA_T, )
    print("="*80)
    print(f"Tempo CPU (NumPy)          : {tempo_cpu:.4f} segundos")
    print("-" * 80)
    print("GPU (Naive):")
    print(f"  - Tempo de execução      : {tempo_gpu_naive:.4f} segundos")
    print(f"  - Speedup vs CPU         : {tempo_cpu / tempo_gpu_naive:.2f}x mais rápido")
    print(f"  - Desvio numérico vs CPU : {desvio_cpu_vs_gpu_naive:.6f}")
    print("-" * 80)
    print("GPU (Shared Memory):")
    print(f"  - Tempo de execução      : {tempo_gpu_opt:.4f} segundos")
    print(f"  - Speedup vs CPU         : {tempo_cpu / tempo_gpu_opt:.2f}x mais rápido")
    print(f"  - Desvio numérico vs CPU : {desvio_cpu_vs_gpu_opt:.6f}")
    print("-" * 80)
    print("GPU (Float4 Vector):")
    print(f"  - Tempo de execução      : {tempo_gpu_vec:.4f} segundos")
    print(f"  - Speedup vs CPU         : {tempo_cpu / tempo_gpu_vec:.2f}x mais rápido")
    print(f"  - Desvio numérico vs CPU : {desvio_cpu_vs_gpu_vec:.6f}")
    print("="*80)
    






    # --- PROVA VISUAL (Matplotlib) - PODE SER LENTO COM MUITAS PARTICULAS/PASSOS. ---
    DESENHAR = False  # True para gerar gráficos
    if DESENHAR == True:
        print("\n-> A re-executar simulação na CPU para gerar o histórico de posições para o gráfico...")
        # Para desenhar, precisamos do histórico. Re-executamos a simulação na CPU,
        # mas desta vez com `guardar_historico=True`.
        pos_para_grafico = posicoes.copy()
        vel_para_grafico = velocidades.copy()
        hist_cpu_para_grafico = simular_n_corpos_cpu(pos_para_grafico, vel_para_grafico, massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, guardar_historico=True)
        desenhar_grafico_n_corpos(hist_cpu_para_grafico, massas, titulo=f"Dinâmica Orbital: {N_PARTICULAS} Corpos")
    else:
        print("\n-> Visualização desativada para benchmarks puros. Defina DESENHAR=True para gerar gráficos.")

if __name__ == "__main__":
    main()
    