import time
from utilidades import gerar_condicoes_iniciais, desenhar_grafico_n_corpos
from simulacao_cpu import simular_n_corpos_cpu
from simulacao_gpu import simular_n_corpos_gpu
import numpy as np

def main():
    N_PARTICULAS = 1000  # Aumenta isto para 2000 quando fores testar para o relatório
    DELTA_T = 0.01
    PASSOS_TEMPO = 100   # Mantém baixo (100 a 500) para não morreres à espera do CPU
    G = 1.0
    EPSILON = 1.0
    TAMANHO_CAIXA = 100.0   
    
    print("="*50)
    print(f" BENCHMARK N-CORPOS: {N_PARTICULAS} Partículas, {PASSOS_TEMPO} Passos, Δt={DELTA_T}")
    print("="*50)
    
    # Fixar a semente para garantir condições iniciais reproduzíveis para benchmarking
    np.random.seed(42) 

    # Geração dos tensores com base nos parâmetros fixados
    print(f"-> A gerar {N_PARTICULAS} partículas num espaço {TAMANHO_CAIXA}^3...")
    massas, posicoes, velocidades = gerar_condicoes_iniciais(
        N_PARTICULAS, TAMANHO_CAIXA, massa_min=10.0, massa_max=50.0
    )

    def reportar_progresso(passo_atual, total_passos):
        percentagem = (passo_atual / total_passos) * 100
        print(f"   [CPU] Progresso da integração: {percentagem:3.0f}% concluído ({passo_atual}/{total_passos} passos)")
    
    # Copiar os arrays iniciais para garantir que a CPU não altera o estado inicial da GPU
    pos_cpu_init = posicoes.copy()
    vel_cpu_init = velocidades.copy()
    pos_gpu_init = posicoes.copy()
    vel_gpu_init = velocidades.copy()

    print("[1/2] A executar na CPU (NumPy)...")
    inicio_cpu = time.perf_counter()
    hist_cpu = simular_n_corpos_cpu(pos_cpu_init, vel_cpu_init, massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, callback_progresso=reportar_progresso)
    tempo_cpu = time.perf_counter() - inicio_cpu
    pos_final_cpu = pos_cpu_init
    
    print("[2/2] A executar na GPU (PyCUDA)...")
    pos_final_gpu, tempo_gpu = simular_n_corpos_gpu(pos_gpu_init, vel_gpu_init, massas, PASSOS_TEMPO, DELTA_T, G, EPSILON)

    # 4. VALIDAÇÃO MATEMÁTICA E RESULTADOS
    # Usamos np.allclose porque operações float32 em C++ e NumPy vão divergir ligeiramente 
    # na ordem das décimas/centésimas após centenas de iterações.
    valido = np.allclose(pos_final_cpu, pos_final_gpu, atol=1e-1)

    print("\n" + "="*50)
    print(" RESULTADOS DO BENCHMARK")
    print("="*50)
    print(f"Tempo CPU : {tempo_cpu:.4f} segundos")
    print(f"Tempo GPU : {tempo_gpu:.4f} segundos")
    print(f"Speedup   : {tempo_cpu / tempo_gpu:.2f}x mais rápido")
    print(f"Validação : {'✅ SUCESSO (Cálculos convergem)' if valido else '❌ FALHA (Desvio matemático excessivo)'}")
    print("="*50)
    
    # Prova Visual (Matplotlib) - pode ser lento com muitas partículas/passos.
    # É boa ideia só desenhar um subconjunto ou desativar para benchmarks puros.
    DESENHAR = True  # Definir como True para visualizar, mas cuidado com o tempo de renderização!
    if N_PARTICULAS <= 64 and DESENHAR:
        print("-> A compilar projeção 2D das órbitas...")
        desenhar_grafico_n_corpos(hist_cpu, titulo=f"Dinâmica Orbital: {N_PARTICULAS} Corpos", N = N_PARTICULAS)
    else:
        print("-> Geração do gráfico desativada para N > 64 (pode ser muito lenta).")
        if not DESENHAR:
            print("   (A variável 'DESENHAR' está definida como False em main.py)")
        else:
            print("   (Altere a variável 'desenhar' em main.py para forçar a visualização)")

if __name__ == "__main__":
    main()