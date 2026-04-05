import time
from utilidades import gerar_condicoes_iniciais, desenhar_grafico_n_corpos
from simulacao_cpu import simular_n_corpos_cpu
from simulacao_gpu import simular_n_corpos_gpu
import numpy as np

def main():
    N_PARTICULAS = 500       # Aumentado para saturar a GPU (ex: 8192, 16384, 32768)
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
    pos_cpu_init, pos_gpu_init = posicoes.copy(), posicoes.copy()
    vel_cpu_init, vel_gpu_init = velocidades.copy(), velocidades.copy()

    # -- 1. CPU SIMULAÇÃO --
    print("[1/4] A executar na CPU (NumPy)...")
    def reportar_progresso(passo_atual, total_passos):
        percentagem = (passo_atual / total_passos) * 100
        print(f"   [CPU] Progresso da integração: {percentagem:3.0f}% concluído ({passo_atual}/{total_passos} passos)")

    inicio_cpu = time.perf_counter()
    hist_cpu = simular_n_corpos_cpu(pos_cpu_init, vel_cpu_init, massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, callback_progresso=None)  # Desativar callback para benchmarks puros
    # Para benchmarks rápidos, podemos desativar o histórico e o callback
    # hist_cpu = simular_n_corpos_cpu(pos_cpu_init, vel_cpu_init, massas, PASSOS_TEMPO, DELTA_T, G, EPSILON)
    tempo_cpu = time.perf_counter() - inicio_cpu
    pos_final_cpu = pos_cpu_init # A posição final é atualizada in-place, então pos_cpu_init já contém o resultado final
    
    # -- 2. GPU SIMULAÇÃO (BASELINE) --
    print("[2/4] A executar na GPU (PyCUDA - Baseline)...")
    pos_final_gpu_base, tempo_gpu_base = simular_n_corpos_gpu(pos_gpu_init.copy(), vel_gpu_init.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='naive')

    # -- 3. GPU SIMULAÇÃO (OTIMIZADA) --
    print("[3/4] A executar na GPU (PyCUDA - Otimizado com Shared Memory)...")
    pos_final_gpu_opt, tempo_gpu_opt = simular_n_corpos_gpu(pos_gpu_init.copy(), vel_gpu_init.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='shared_mem')

    # -- 4. GPU SIMULAÇÃO (VECTOR TYPES) --
    print("[4/4] A executar na GPU (PyCUDA - Memória de Banda Larga float4)...")
    pos_final_gpu_vec, tempo_gpu_vec = simular_n_corpos_gpu(pos_gpu_init.copy(), vel_gpu_init.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='shared_mem_float4')

    # VALIDAÇÃO MATEMÁTICA E RESULTADOS
    # O desvio máximo entre as posições finais da CPU e GPU deve ser pequeno (dependendo do softening e do número de passos).
    desvio_cpu_vs_gpu_naive = np.max(np.abs(pos_final_cpu - pos_final_gpu_base))
    desvio_naive_vs_vec = np.max(np.abs(pos_final_gpu_base - pos_final_gpu_vec))
    desvio_opt_vs_vec = np.max(np.abs(pos_final_gpu_opt - pos_final_gpu_vec))

    print("\n" + "="*80)
    print(" RESULTADOS DO BENCHMARK para" ,N_PARTICULAS, "partículas,", PASSOS_TEMPO, "passos, Δt=", DELTA_T, )
    print("="*80)
    print(f"Tempo CPU (NumPy)          : {tempo_cpu:.4f} segundos")
    print(f"Tempo GPU (Naive)          : {tempo_gpu_base:.4f} segundos")
    print(f"Tempo GPU (Shared Memory)  : {tempo_gpu_opt:.4f} segundos")
    print(f"Tempo GPU (Float4 Vector)  : {tempo_gpu_vec:.4f} segundos")
    print("-" * 80)
    print(f"Speedup (CPU vs GPU Float4): {tempo_cpu / tempo_gpu_vec:.2f}x mais rápido")
    print(f"Speedup (GPU Opt vs Naive) : {tempo_gpu_base / tempo_gpu_opt:.2f}x mais rápido")
    print(f"Speedup Float4 (Vec vs Opt): {tempo_gpu_opt / tempo_gpu_vec:.2f}x mais rápido (Ganho de Memória!)")
    print("-" * 80)
    print("ANÁLISE DE DESVIO NUMÉRICO:")
    print(f"Desvio (CPU vs GPU Naive)      : {desvio_cpu_vs_gpu_naive:.6f} (Validação de correção. Deve ser ~0)")
    print(f"Desvio (Naive vs Float4 Vector): {desvio_naive_vs_vec:.6f} (Impacto da otimização rsqrtf acumulado)")
    print(f"Desvio (Shared Mem vs Float4)  : {desvio_opt_vs_vec:.6f} (Deve ser 0, usam a exata mesma matemática)")
    print("="*80)
    





    # # --- ANÁLISE DE DIVERGÊNCIA EM PASSO ÚNICO ---
    # print("\n" + "="*80)
    # print(" ANÁLISE DE DIVERGÊNCIA EM PASSO ÚNICO (Single-Step Validation)")
    # print("="*80)
    # print("A executar novamente com PASSOS_TEMPO=1 para isolar o erro numérico por passo...")

    # # Re-executar as simulações por apenas um passo para medir o erro por iteração
    # pos_cpu_1step = posicoes.copy()
    # vel_cpu_1step = velocidades.copy()
    # simular_n_corpos_cpu(pos_cpu_1step, vel_cpu_1step, massas, 1, DELTA_T, G, EPSILON)

    # pos_gpu_naive_1step, _ = simular_n_corpos_gpu(pos_gpu_init.copy(), vel_gpu_init.copy(), massas, 1, DELTA_T, G, EPSILON, method='naive')
    # pos_gpu_opt_1step, _ = simular_n_corpos_gpu(pos_gpu_init.copy(), vel_gpu_init.copy(), massas, 1, DELTA_T, G, EPSILON, method='shared_mem')
    # pos_gpu_vec_1step, _ = simular_n_corpos_gpu(pos_gpu_init.copy(), vel_gpu_init.copy(), massas, 1, DELTA_T, G, EPSILON, method='shared_mem_float4')

    # desvio_1step_cpu_vs_naive = np.max(np.abs(pos_cpu_1step - pos_gpu_naive_1step))
    # desvio_1step_naive_vs_vec = np.max(np.abs(pos_gpu_naive_1step - pos_gpu_vec_1step))

    # print(f"Desvio (CPU vs Naive) após 1 passo: {desvio_1step_cpu_vs_naive:.10f}")
    # print(f"Erro por passo (Naive vs Float4) : {desvio_1step_naive_vs_vec:.10f} <-- Erro da instrução rsqrtf documentado!")
    # print("="*80)


    # --- PROVA VISUAL (Matplotlib) - PODE SER LENTO COM MUITAS PARTICULAS/PASSOS. ---
    DESENHAR = False  # True para gerar gráficos
    if DESENHAR == True:
        print("-> A gerar projeção 2D das órbitas (tamanho do ponto proporcional à massa)...")
        desenhar_grafico_n_corpos(hist_cpu, massas, titulo=f"Dinâmica Orbital: {N_PARTICULAS} Corpos")
    else:
        print("-> Visualização desativada para benchmarks puros. Defina DESENHAR=True para gerar gráficos.")

if __name__ == "__main__":
    main()
    