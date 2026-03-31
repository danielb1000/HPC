import time
from utilidades import gerar_condicoes_iniciais, desenhar_grafico_n_corpos
from simulacao_cpu import simular_n_corpos_cpu
import numpy as np

def main():
    # --- PARÂMETROS DA SIMULAÇÃO (Otimizados para Benchmark) ---
    # Número de corpos. A complexidade é O(N^2), por isso este é o fator mais
    # importante para o tempo de execução. Valores como 256, 512, 1024, 2048
    # são bons para testar e ver a diferença entre CPU e GPU.
    N_PARTICULAS = 1024

    # Tamanho do passo temporal (dt). Crítico para a estabilidade da simulação.
    # Se as partículas "explodirem" e saírem do ecrã, reduza este valor.
    DELTA_T = 0.01

    # Número de passos de tempo. Para benchmarking, não precisa de ser um valor
    # muito alto. O suficiente para obter uma medição de tempo estável.
    PASSOS_TEMPO = 1000

    # Constante Gravitacional. Usar 1.0 é uma prática comum (unidades normalizadas)
    # para manter os valores numéricos bem comportados e evitar erros de precisão.
    G = 1.0

    # Fator de suavização. Evita a singularidade (força infinita) quando duas
    # partículas se aproximam muito. Previne instabilidades numéricas.
    EPSILON = 1.0

    # Volume inicial onde as partículas são geradas.
    TAMANHO_CAIXA = 100.0    
    
    print("="*50)
    print(" SIMULADOR N-CORPOS: Avaliação CPU (NumPy)")
    print("="*50)
    
    # Fixar a semente para garantir condições iniciais reproduzíveis para benchmarking
    np.random.seed(42) 

    # Geração dos tensores com base nos parâmetros fixados
    print(f"-> A gerar {N_PARTICULAS} partículas num espaço {TAMANHO_CAIXA}^3...")
    massas, posicoes, velocidades = gerar_condicoes_iniciais(
        N_PARTICULAS, 
        TAMANHO_CAIXA, 
        massa_min=10.0, 
        massa_max=50.0
    )

    def reportar_progresso(passo_atual, total_passos):
        percentagem = (passo_atual / total_passos) * 100
        print(f"   [CPU] Progresso da integração: {percentagem:3.0f}% concluído ({passo_atual}/{total_passos} passos)")
    
    print(f"-> A iniciar integração temporal (Velocity Verlet) para {PASSOS_TEMPO} passos...")
    tempo_inicio = time.time()
    
    # Motor de Integração O(N^2)
    historico = simular_n_corpos_cpu(
        posicoes, velocidades, massas, 
        PASSOS_TEMPO, DELTA_T, G, EPSILON,
        callback_progresso=reportar_progresso
    )
    
    tempo_fim = time.time()
    tempo_execucao = tempo_fim - tempo_inicio
    
    # Métricas de desempenho
    print(f"-> Concluído em {tempo_execucao:.4f} segundos.")
    interacoes_por_segundo = (N_PARTICULAS * N_PARTICULAS * PASSOS_TEMPO) / tempo_execucao
    print(f"   Desempenho: {interacoes_por_segundo / 1e6:.2f} M-interações/segundo")
    print("="*50)
    
    # Prova Visual (Matplotlib) - pode ser lento com muitas partículas/passos.
    # É boa ideia só desenhar um subconjunto ou desativar para benchmarks puros.
    DESENHAR = True  # Definir como True para visualizar, mas cuidado com o tempo de renderização!
    if N_PARTICULAS <= 64 and DESENHAR:
        print("-> A compilar projeção 2D das órbitas...")
        desenhar_grafico_n_corpos(historico, titulo=f"Dinâmica Orbital: {N_PARTICULAS} Corpos", N = N_PARTICULAS)
    else:
        print("-> Geração do gráfico desativada para N > 64 (pode ser muito lenta).")
        if not DESENHAR:
            print("   (A variável 'DESENHAR' está definida como False em main.py)")
        else:
            print("   (Altere a variável 'desenhar' em main.py para forçar a visualização)")

if __name__ == "__main__":
    main()