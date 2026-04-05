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
    print("[1/3] A executar na CPU (NumPy)...")
    def reportar_progresso(passo_atual, total_passos):
        percentagem = (passo_atual / total_passos) * 100
        print(f"   [CPU] Progresso da integração: {percentagem:3.0f}% concluído ({passo_atual}/{total_passos} passos)")

    inicio_cpu = time.perf_counter()
    hist_cpu = simular_n_corpos_cpu(pos_cpu_init, vel_cpu_init, massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, callback_progresso=reportar_progresso)
    # Para benchmarks rápidos, podemos desativar o histórico e o callback
    # hist_cpu = simular_n_corpos_cpu(pos_cpu_init, vel_cpu_init, massas, PASSOS_TEMPO, DELTA_T, G, EPSILON)
    tempo_cpu = time.perf_counter() - inicio_cpu
    pos_final_cpu = pos_cpu_init # A posição final é atualizada in-place, então pos_cpu_init já contém o resultado final
    
    # -- 2. GPU SIMULAÇÃO (BASELINE) --
    print("[2/3] A executar na GPU (PyCUDA - Baseline)...")
    pos_final_gpu_base, tempo_gpu_base = simular_n_corpos_gpu(pos_gpu_init.copy(), vel_gpu_init.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='naive')

    # -- 3. GPU SIMULAÇÃO (OTIMIZADA) --
    print("[3/3] A executar na GPU (PyCUDA - Otimizado com Shared Memory)...")
    pos_final_gpu_opt, tempo_gpu_opt = simular_n_corpos_gpu(pos_gpu_init.copy(), vel_gpu_init.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='shared_mem')

    # VALIDAÇÃO MATEMÁTICA E RESULTADOS
    # O desvio máximo entre as posições finais da CPU e GPU deve ser pequeno (dependendo do softening e do número de passos).
    desvio_cpu_vs_gpu_opt = np.max(np.abs(pos_final_cpu - pos_final_gpu_opt))
    desvio_gpu_vs_gpu = np.max(np.abs(pos_final_gpu_base - pos_final_gpu_opt))

    print("\n" + "="*80)
    print(" RESULTADOS DO BENCHMARK para" ,N_PARTICULAS, "partículas,", PASSOS_TEMPO, "passos, Δt=", DELTA_T, )
    print("="*80)
    print(f"Tempo CPU (NumPy)          : {tempo_cpu:.4f} segundos")
    print(f"Tempo GPU (Naive)          : {tempo_gpu_base:.4f} segundos")
    print(f"Tempo GPU (Shared Memory)  : {tempo_gpu_opt:.4f} segundos")
    print(f"Speedup (CPU vs GPU Opt)   : {tempo_cpu / tempo_gpu_opt:.2f}x mais rápido")
    print(f"Speedup (GPU Opt vs Naive) : {tempo_gpu_base / tempo_gpu_opt:.2f}x mais rápido")
    print(f"Desvio Máximo (CPU vs GPU Opt) : {desvio_cpu_vs_gpu_opt:.6f}")
    print(f"Desvio Máximo (GPU vs GPU Opt) : {desvio_gpu_vs_gpu:.6f} (deve ser próximo de zero)")
    print("="*80)
    
    # Prova Visual (Matplotlib) - pode ser lento com muitas partículas/passos.
    # É boa ideia só desenhar um subconjunto ou desativar para benchmarks puros.
    DESENHAR = False  # True para gerar gráficos
    if DESENHAR == True:
        print("-> A gerar projeção 2D das órbitas (tamanho do ponto proporcional à massa)...")
        desenhar_grafico_n_corpos(hist_cpu, massas, titulo=f"Dinâmica Orbital: {N_PARTICULAS} Corpos")
    else:
        print("-> Visualização desativada para benchmarks puros. Defina DESENHAR=True para gerar gráficos.")

if __name__ == "__main__":
    main()
    