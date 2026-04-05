import time
from utilidades import gerar_condicoes_iniciais, desenhar_grafico_n_corpos
from simulacao_cpu import simular_n_corpos_cpu
from simulacao_gpu import simular_n_corpos_gpu
import numpy as np

def main():
    N_PARTICULAS = 100       # Aumentado para saturar a GPU (ex: 8192, 16384, 32768)
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
    massas, posicoes, velocidades = gerar_condicoes_iniciais(N_PARTICULAS, TAMANHO_CAIXA, MASSA_MIN, MASSA_MAX, VELOCIDADE_MIN,VELOCIDADE_MAX, )

    # Criar cópias para CPU e GPU para garantir que ambos começam com as mesmas condições iniciais
    pos_cpu_init, pos_gpu_init = posicoes.copy(), posicoes.copy()
    vel_cpu_init, vel_gpu_init = velocidades.copy(), velocidades.copy()

    print("[1/2] A executar na CPU (NumPy)...")
    def reportar_progresso(passo_atual, total_passos):
        percentagem = (passo_atual / total_passos) * 100
        print(f"   [CPU] Progresso da integração: {percentagem:3.0f}% concluído ({passo_atual}/{total_passos} passos)")

    inicio_cpu = time.perf_counter()
    hist_cpu = simular_n_corpos_cpu(pos_cpu_init, vel_cpu_init, massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, callback_progresso=reportar_progresso)
    tempo_cpu = time.perf_counter() - inicio_cpu
    pos_final_cpu = pos_cpu_init # A posição final é atualizada in-place, então pos_cpu_init já contém o resultado final
    
    print("[2/2] A executar na GPU (PyCUDA)...")
    pos_final_gpu, tempo_gpu = simular_n_corpos_gpu(pos_gpu_init, vel_gpu_init, massas, PASSOS_TEMPO, DELTA_T, G, EPSILON)

    # VALIDAÇÃO MATEMÁTICA E RESULTADOS
    # O desvio máximo entre as posições finais da CPU e GPU deve ser pequeno (dependendo do softening e do número de passos).
    desvio_maximo = np.max(np.abs(pos_final_cpu - pos_final_gpu))

    print("\n" + "="*80)
    print(" RESULTADOS DO BENCHMARK para" ,N_PARTICULAS, "partículas,", PASSOS_TEMPO, "passos, Δt=", DELTA_T, )
    print("="*80)
    print(f"Tempo CPU : {tempo_cpu:.4f} segundos")
    print(f"Tempo GPU : {tempo_gpu:.4f} segundos")
    print(f"Speedup   : {tempo_cpu / tempo_gpu:.2f}x mais rápido")
    print(f"Desvio Máximo (CPU vs GPU): {desvio_maximo:.6f} (tolerância mínima para convergência)")
    print("="*80)
    
    # Prova Visual (Matplotlib) - pode ser lento com muitas partículas/passos.
    # É boa ideia só desenhar um subconjunto ou desativar para benchmarks puros.
    DESENHAR = False  # Definir como True para visualizar, mas cuidado com o tempo de renderização! # False no servidor.
    if N_PARTICULAS <= 64 or DESENHAR:
        print("-> A gerar projeção 2D das órbitas (tamanho do ponto proporcional à massa)...")
        desenhar_grafico_n_corpos(hist_cpu, massas, titulo=f"Dinâmica Orbital: {N_PARTICULAS} Corpos")
    else:
        print("-> Geração do gráfico desativada para N > 64 (pode ser muito lenta).")
        if not DESENHAR:
            print("   (A variável 'DESENHAR' está definida como False em main.py)")
        else:
            print("   (Altere a variável 'desenhar' em main.py para forçar a visualização)")

if __name__ == "__main__":
    main()
    