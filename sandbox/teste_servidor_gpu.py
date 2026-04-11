import multiprocessing as mp
import numpy as np

# Este script é um teste de diagnóstico para verificar se o servidor com 4 GPUs está configurado corretamente e pode ser acessado via PyCUDA.

def testar_gpu(gpu_id, queue):
    """
    Função executada por cada processo isolado. 
    Inicializa o driver CUDA apenas para a sua GPU designada e faz um teste de memória.
    """
    import pycuda.driver as cuda
    
    try:
        cuda.init()
        dev = cuda.Device(gpu_id)
        ctx = dev.make_context()
        
        nome = dev.name()
        mem_livre, mem_total = cuda.mem_get_info()
        
        # Teste de alocação de memória e transferência HtoD (Host-to-Device)
        a = np.random.randn(1000).astype(np.float32)
        a_gpu = cuda.mem_alloc(a.nbytes)
        cuda.memcpy_htod(a_gpu, a)
        
        resultado = f"GPU {gpu_id}: {nome} | Mem.: {mem_livre / 1024**3:.2f}GB livres de {mem_total / 1024**3:.2f}GB. Teste OK!"
        
        # Limpar contexto (boa prática em HPC)
        ctx.pop()
        queue.put((gpu_id, resultado))
        
    except Exception as e:
        queue.put((gpu_id, f"GPU {gpu_id}: ERRO - {str(e)}"))

if __name__ == "__main__":
    # Crucial em Linux ao usar CUDA com multiprocessing para evitar crashes de driver
    mp.set_start_method('spawn') 
    
    # Tentar usar as 4 GPUs do servidor
    num_gpus_alvo = 4
    queue = mp.Queue()
    processos = []
    
    print(f"A iniciar teste distribuído em {num_gpus_alvo} GPUs...")
    for i in range(num_gpus_alvo):
        p = mp.Process(target=testar_gpu, args=(i, queue))
        p.start()
        processos.append(p)
        
    for p in processos:
        p.join()
        
    resultados = [queue.get() for _ in range(num_gpus_alvo)]
    resultados.sort(key=lambda x: x[0])
    
    print("\n" + "="*60)
    print(" RESULTADOS DO DIAGNÓSTICO DO SERVIDOR")
    print("="*60)
    for r in resultados:
        print(r[1])
    print("="*60)
