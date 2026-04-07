import time
import multiprocessing as mp
from utilidades import gerar_condicoes_iniciais, desenhar_grafico_n_corpos
from simulacao_cpu import simular_n_corpos_cpu
from simulacao_gpu import simular_n_corpos_gpu, validar_energia_gpu
from simulacao_multigpu import simular_n_corpos_multigpu
from simulacao_nv4 import simular_n_corpos_nv4
import numpy as np
import pycuda.driver as cuda


def main():
    N_PARTICULAS = 100    # Aumentado para saturar a GPU (ex: 8192, 16384, 32768)
    DELTA_T = 0.01         # Tempo de passo
    PASSOS_TEMPO = 200     # Reduzido para evitar que o CPU demore demasiadas horas
    TAMANHO_CAIXA = np.cbrt(N_PARTICULAS / 0.002) # Calcular tamanho da caixa dinamicamente para manter densidade constante (~0.002). 
    EPSILON = 1.0           # Softening parameter = 1.0 em vez de 0.1 para evitar divergências numéricas em simulações longas.
    G = 1.0                 # Constante gravitacional
    MASSA_MIN = 10.0        # Massa mínima das partículas
    MASSA_MAX = 50.0        # Massa máxima das partículas
    VELOCIDADE_MIN = -1.0   # Velocidade mínima das partículas
    VELOCIDADE_MAX = 1.0    # Velocidade máxima das partículas
    np.random.seed(42)      # Seed fixa para resultados consistentes

    IGNORAR_CPU = False  # True para benchmarks puros focados na GPU, False para comparação CPU vs GPU
    cuda.init()
    num_gpus = cuda.Device.count()

    print("="*75)
    print(f" BENCHMARK N-CORPOS: {N_PARTICULAS} Partículas, {PASSOS_TEMPO} Passos, Δt={DELTA_T}, Espaço = {TAMANHO_CAIXA:.2f}^3")
    print("="*75)
    print(f" GPUs detetadas no sistema: {num_gpus}")
    print(f" Dispositivo Principal (GPU 0): {cuda.Device(0).name()}\n")
    


    # -------------------------------------------------------------------------
    # GESTÃO DE CONTEXTO MANUAL: Substitui o pycuda.autoinit
    # Criamos o contexto na GPU 0 para ser usado pela CPU e simulações single-GPU
    # -------------------------------------------------------------------------
    ctx = cuda.Device(0).make_context()
    
    try:
        # Geração dos tensores (NumPy) de massas, posições e velocidades iniciais
        massas, posicoes, velocidades = gerar_condicoes_iniciais(
            N_PARTICULAS, TAMANHO_CAIXA, MASSA_MIN, MASSA_MAX, VELOCIDADE_MIN,VELOCIDADE_MAX, 
            )

        # Criar cópias para CPU e GPU para garantir que ambos começam com as mesmas condições iniciais
        pos_para_cpu = posicoes.copy()
        vel_para_cpu = velocidades.copy()

        # -- 0. VALIDAÇÃO FÍSICA (GPU) --
        print("[0/6] A calcular Energia Total Inicial na GPU...")
        energia_inicial_gpu = validar_energia_gpu(posicoes, velocidades, massas, G, EPSILON)

        # -- 1. CPU SIMULAÇÃO --
        if not IGNORAR_CPU:
            print("[1/6] A executar na CPU (NumPy)...")
            inicio_cpu = time.perf_counter()
            simular_n_corpos_cpu(pos_para_cpu, vel_para_cpu, massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, guardar_historico=False)
            tempo_cpu = time.perf_counter() - inicio_cpu
            pos_final_cpu = pos_para_cpu  
        else:
            print("[1/6] A executar na CPU (NumPy)... IGNORADO (IGNORAR_CPU = True)")
            tempo_cpu = None
            pos_final_cpu = None
        
        # -- 2. GPU SIMULAÇÃO (BASELINE (NAIVE)) --
        print("[2/6] A executar na GPU (PyCUDA - Baseline (Naive))...")
        pos_final_gpu_naive, vel_final_gpu_naive, tempo_gpu_naive = simular_n_corpos_gpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='naive')

        # -- 3. GPU SIMULAÇÃO (NAIVE + FAST MATH) --
        print("[3/6] A executar na GPU (PyCUDA - Naive + Fast Math)...")
        pos_final_gpu_naive_fm, vel_final_gpu_naive_fm, tempo_gpu_naive_fm = simular_n_corpos_gpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='naive_fast_math')

        # -- 4. GPU SIMULAÇÃO (SHARED MEMORY + FAST MATH) --
        print("[4/6] A executar na GPU (PyCUDA - Shared Memory + Fast Math)...")
        pos_final_gpu_opt, vel_final_gpu_opt, tempo_gpu_opt = simular_n_corpos_gpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='shared_mem')

        # -- 5. GPU SIMULAÇÃO (SHARED MEMORY + FLOAT4 + FAST MATH) --
        print("[5/6] A executar na GPU (PyCUDA - Shared Mem + Float4 Vector + Fast Math)...")
        pos_final_gpu_vec, vel_final_gpu_vec, tempo_gpu_vec = simular_n_corpos_gpu(posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, method='shared_mem_float4')

        # -- 6. GPU SIMULAÇÃO (MULTI-GPU) --
        tempo_multigpu = None
        desvio_cpu_vs_multigpu = None
        if num_gpus > 1:
            print(f"[6/6] A executar na GPU (Multi-GPU Distribuído - {num_gpus} GPUs)...")
            # Para o script multi-gpu correr em segurança, ele não partilha este contexto CUDA principal. 
            # Ele cria os próprios contextos nos workers isolados.
            pos_final_multigpu, vel_final_multigpu, tempo_multigpu = simular_n_corpos_multigpu(
                posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, num_gpus_ativas=num_gpus
            )
        else:
            print("[6/6] Ignorado. Sistema tem apenas 1 GPU.")

        # -- 7. GPU SIMULAÇÃO (NVLINK / NVSWITCH P2P) --
        tempo_nv4 = None
        if num_gpus > 1:
            print(f"[7/7] A executar na GPU (NVLink/NVSwitch P2P - {num_gpus} GPUs)...")
            pos_final_nv4, vel_final_nv4, tempo_nv4 = simular_n_corpos_nv4(
                posicoes.copy(), velocidades.copy(), massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, num_gpus_ativas=num_gpus
            )

        # VALIDAÇÃO MATEMÁTICA E RESULTADOS
        # Se não houver CPU, usamos a versão Naive como a fonte da verdade para desvios numéricos
        ref_nome = "CPU" if not IGNORAR_CPU else "GPU Naive"
        pos_ref = pos_final_cpu if not IGNORAR_CPU else pos_final_gpu_naive

        desvio_naive = np.max(np.abs(pos_ref - pos_final_gpu_naive))
        desvio_naive_fm = np.max(np.abs(pos_ref - pos_final_gpu_naive_fm))
        desvio_opt = np.max(np.abs(pos_ref - pos_final_gpu_opt))
        desvio_vec = np.max(np.abs(pos_ref - pos_final_gpu_vec))

        print("A calcular Energia Total Final na GPU...")
        energia_final_gpu = validar_energia_gpu(pos_final_gpu_vec, vel_final_gpu_vec, massas, G, EPSILON)
        erro_energia = abs((energia_final_gpu - energia_inicial_gpu) / energia_inicial_gpu) * 100
        
        str_tempo_cpu = f"{tempo_cpu:.4f} segundos" if not IGNORAR_CPU else "IGNORADO"
        def spd(t): return f"{tempo_cpu / t:.2f}x mais rápido" if not IGNORAR_CPU else "N/A"

        print("\n" + "="*80)
        print(" RESULTADOS DO BENCHMARK para" ,N_PARTICULAS, "partículas,", PASSOS_TEMPO, "passos, Δt=", DELTA_T, )
        print("="*80)
        print(f"Tempo CPU (NumPy)          : {str_tempo_cpu}")
        print("-" * 80)
        print("GPU (Naive):")
        print(f"  - Tempo de execução      : {tempo_gpu_naive:.4f} segundos")
        print(f"  - Speedup vs CPU         : {spd(tempo_gpu_naive)}")
        print(f"  - Desvio numérico vs {ref_nome} : {desvio_naive:.6f}")
        print("-" * 80)
        print("GPU (Naive + Fast Math):")
        print(f"  - Tempo de execução      : {tempo_gpu_naive_fm:.4f} segundos")
        print(f"  - Speedup vs CPU         : {spd(tempo_gpu_naive_fm)}")
        print(f"  - Desvio numérico vs {ref_nome} : {desvio_naive_fm:.6f}")
        print("-" * 80)
        print("GPU (Shared Memory + Fast Math):")
        print(f"  - Tempo de execução      : {tempo_gpu_opt:.4f} segundos")
        print(f"  - Speedup vs CPU         : {spd(tempo_gpu_opt)}")
        print(f"  - Desvio numérico vs {ref_nome} : {desvio_opt:.6f}")
        print("-" * 80)
        print("GPU (Shared Mem + Float4 + Fast Math):")
        print(f"  - Tempo de execução      : {tempo_gpu_vec:.4f} segundos")
        print(f"  - Speedup vs CPU         : {spd(tempo_gpu_vec)}")
        print(f"  - Desvio numérico vs {ref_nome} : {desvio_vec:.6f}")
        
        if num_gpus > 1:
            desvio_multigpu = np.max(np.abs(pos_ref - pos_final_multigpu))
            print("-" * 80)
            print(f"MULTI-GPU ({num_gpus} Placas):")
            print(f"  - Tempo de execução      : {tempo_multigpu:.4f} segundos")
            print(f"  - Speedup vs CPU         : {spd(tempo_multigpu)}")
            print(f"  - Speedup vs GPU Float4  : {tempo_gpu_vec / tempo_multigpu:.2f}x mais rápido")
            print(f"  - Desvio numérico vs {ref_nome} : {desvio_multigpu:.6f}")
            
        if num_gpus > 1 and tempo_nv4 is not None:
            desvio_nv4 = np.max(np.abs(pos_ref - pos_final_nv4))
            print("-" * 80)
            print(f"MULTI-GPU NVLINK/NVSWITCH ({num_gpus} Placas P2P):")
            print(f"  - Tempo de execução      : {tempo_nv4:.4f} segundos")
            print(f"  - Speedup vs CPU         : {spd(tempo_nv4)}")
            print(f"  - Speedup vs Multi-GPU   : {tempo_multigpu / tempo_nv4:.2f}x mais rápido")
            print(f"  - Desvio numérico vs {ref_nome} : {desvio_nv4:.6f}")

        print("="*80)
        print(" VALIDAÇÃO FÍSICA (Conservação de Energia em CUDA)")
        print(f"  - Energia Total Inicial  : {energia_inicial_gpu:.2f} J")
        print(f"  - Energia Total Final    : {energia_final_gpu:.2f} J")
        print(f"  - Erro de Conservação    : {erro_energia:.6f}%")
        print("="*80)

        # --- PROVA VISUAL (Matplotlib) ---
        DESENHAR = True  # True para gerar gráficos
        if DESENHAR == True and not IGNORAR_CPU:
            print("\n-> A re-executar simulação na CPU para gerar o histórico de posições para o gráfico...")
            pos_para_grafico = posicoes.copy()
            vel_para_grafico = velocidades.copy()
            hist_cpu_para_grafico = simular_n_corpos_cpu(pos_para_grafico, vel_para_grafico, massas, PASSOS_TEMPO, DELTA_T, G, EPSILON, guardar_historico=True)
            desenhar_grafico_n_corpos(hist_cpu_para_grafico, massas, titulo=f"Dinâmica Orbital: {N_PARTICULAS} Corpos")
        elif DESENHAR == True and IGNORAR_CPU:
            print("\n-> Visualização ignorada: O Matplotlib precisa do histórico (IGNORAR_CPU está ativo).")
        else:
            print("\n-> Visualização desativada para benchmarks puros. Defina DESENHAR=True para gerar gráficos.")
            
    finally:
        # Limpar o contexto CUDA de forma elegante ao fechar
        ctx.pop()

if __name__ == "__main__":
    # Método de arranque seguro para PyCUDA com Multiprocessing
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
        
    main()