import multiprocessing as mp
import numpy as np
import time
from utils import gerar_condicoes_iniciais

# Kernel exclusivo para Multi-GPU. 
# Nota: Não importamos o pycuda globalmente para evitar o autoinit.
KERNEL_CODE_MULTIGPU = """
#define BLOCK_SIZE 256

__global__ void pre_update_multigpu(float4 *pos_mass_local, float *vel_local, float *acel_local, float dt, int N_local) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N_local) {
        float4 p = pos_mass_local[i];
        
        float v_half_x = vel_local[i*3 + 0] + 0.5f * acel_local[i*3 + 0] * dt;
        float v_half_y = vel_local[i*3 + 1] + 0.5f * acel_local[i*3 + 1] * dt;
        float v_half_z = vel_local[i*3 + 2] + 0.5f * acel_local[i*3 + 2] * dt;

        p.x += v_half_x * dt;
        p.y += v_half_y * dt;
        p.z += v_half_z * dt;

        pos_mass_local[i] = p; // Atualiza a posição local
        
        vel_local[i*3 + 0] = v_half_x; 
        vel_local[i*3 + 1] = v_half_y; 
        vel_local[i*3 + 2] = v_half_z;
    }
}

__global__ void calcular_aceleracoes_multigpu(
    float4 *pos_mass_all, float4 *pos_mass_local, float *acel_local, 
    int N_total, int N_local, int offset, float G, float eps) 
{
    __shared__ float4 s_pos_mass[BLOCK_SIZE];

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    
    float pos_i_x = 0.0f, pos_i_y = 0.0f, pos_i_z = 0.0f;
    int global_i = offset + i; // O verdadeiro ID da partícula no Universo

    if (i < N_local) {
        float4 my_pos = pos_mass_local[i];
        pos_i_x = my_pos.x; pos_i_y = my_pos.y; pos_i_z = my_pos.z;
    }

    int num_tiles = (N_total + blockDim.x - 1) / blockDim.x;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int j_shared_load_idx = tile_idx * blockDim.x + threadIdx.x;
        
        // TODAS as threads ajudam a carregar o mapa GLOBAL para a Shared Memory
        if (j_shared_load_idx < N_total) {
            s_pos_mass[threadIdx.x] = pos_mass_all[j_shared_load_idx];
        } else {
            s_pos_mass[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();

        // APENAS as threads locais validas calculam a sua fatia de aceleração
        if (i < N_local) {
            #pragma unroll
            for (int j_tile = 0; j_tile < blockDim.x; j_tile++) {
                float4 p_j = s_pos_mass[j_tile];
                if (p_j.w > 0.0f && (tile_idx * blockDim.x + j_tile) != global_i) {
                    float dx = p_j.x - pos_i_x; 
                    float dy = p_j.y - pos_i_y; 
                    float dz = p_j.z - pos_i_z;
                    float dist_sq = dx*dx + dy*dy + dz*dz;
                    float inv_dist = rsqrtf(dist_sq + eps*eps);
                    float forca = G * p_j.w * inv_dist * inv_dist * inv_dist;
                    ax += forca * dx; ay += forca * dy; az += forca * dz;
                }
            }
        }
        __syncthreads();
    }
    if (i < N_local) {
        acel_local[i * 3 + 0] = ax; 
        acel_local[i * 3 + 1] = ay; 
        acel_local[i * 3 + 2] = az;
    }
}

__global__ void post_update_multigpu(float *vel_local, float *acel_local, float dt, int N_local) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N_local) {
        for (int d = 0; d < 3; d++) {
            int idx = i * 3 + d;
            vel_local[idx] += 0.5f * acel_local[idx] * dt; 
        }
    }
}
"""

def worker_gpu(gpu_id, num_gpus, N_total, vel_init_local, passos, dt, G, eps, shared_pos_mass_base, barrier_init, barrier_step, queue_result):
    """
    Processo isolado que controla UMA única GPU.
    """
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    
    cuda.init()
    dev = cuda.Device(gpu_id)
    ctx = dev.make_context()
    
    try:
        # Compilar Kernels
        mod = SourceModule(KERNEL_CODE_MULTIGPU, options=["-Xptxas", "-v", "-O3"], cache_dir=None)
        kernel_acel = mod.get_function("calcular_aceleracoes_multigpu")
        kernel_pre = mod.get_function("pre_update_multigpu")
        kernel_post = mod.get_function("post_update_multigpu")
        
        # Definir a sua fatia de trabalho (N_local e Offset)
        N_local_base = N_total // num_gpus
        offset = gpu_id * N_local_base
        N_local = N_local_base if gpu_id < num_gpus - 1 else N_total - offset
        
        # Associar buffer de memória partilhada do CPU a um array NumPy sem cópias adicionais
        pos_mass_all_host = np.frombuffer(shared_pos_mass_base, dtype=np.float32).reshape((N_total, 4))
        pos_mass_local_host = pos_mass_all_host[offset:offset+N_local].copy()
        vel_local_host = vel_init_local.flatten()
        acel_local_host = np.zeros(N_local * 3, dtype=np.float32)

        # Alocar Memória na GPU
        d_pos_mass_all = cuda.mem_alloc(pos_mass_all_host.nbytes)
        d_pos_mass_local = cuda.mem_alloc(pos_mass_local_host.nbytes)
        d_vel_local = cuda.mem_alloc(vel_local_host.nbytes)
        d_acel_local = cuda.mem_alloc(acel_local_host.nbytes)

        # Transferências iniciais Host -> Device
        cuda.memcpy_htod(d_pos_mass_all, pos_mass_all_host)
        cuda.memcpy_htod(d_pos_mass_local, pos_mass_local_host)
        cuda.memcpy_htod(d_vel_local, vel_local_host)

        block_dim = (256, 1, 1)
        grid_dim = (int(np.ceil(N_local / 256)), 1)
        
        # Aceleração Inicial (t=0)
        kernel_acel(d_pos_mass_all, d_pos_mass_local, d_acel_local, np.int32(N_total), np.int32(N_local), np.int32(offset), np.float32(G), np.float32(eps), block=block_dim, grid=grid_dim)
        
        # Sincronizar todas as GPUs antes de arrancar o ciclo de tempo
        cuda.Context.synchronize()
        barrier_init.wait()
        
        for passo in range(passos):
            # 1. Atualizar Posições Locais
            kernel_pre(d_pos_mass_local, d_vel_local, d_acel_local, np.float32(dt), np.int32(N_local), block=block_dim, grid=grid_dim)
            
            # 2. Copiar a fatia atualizada para a Memória Partilhada do Host
            cuda.memcpy_dtoh(pos_mass_local_host, d_pos_mass_local)
            pos_mass_all_host[offset:offset+N_local] = pos_mass_local_host
            
            # SINCRONIZAÇÃO: Esperar que TODAS as 4 GPUs coloquem a sua fatia na RAM do Host
            barrier_step.wait()
            
            # 3. O Host agora tem o universo completo. Puxar o universo completo de volta para a GPU
            cuda.memcpy_htod(d_pos_mass_all, pos_mass_all_host)
            
            barrier_step.wait() # Aguardar que todos leiam do Host antes de alguém o sobrescrever no próximo loop
            
            # 4. Calcular novas forças e atualizar velocidades
            kernel_acel(d_pos_mass_all, d_pos_mass_local, d_acel_local, np.int32(N_total), np.int32(N_local), np.int32(offset), np.float32(G), np.float32(eps), block=block_dim, grid=grid_dim)
            kernel_post(d_vel_local, d_acel_local, np.float32(dt), np.int32(N_local), block=block_dim, grid=grid_dim)

        # Fim da simulação. Resgatar dados finais
        cuda.memcpy_dtoh(pos_mass_local_host, d_pos_mass_local)
        cuda.memcpy_dtoh(vel_local_host, d_vel_local)
        
        queue_result.put((gpu_id, pos_mass_local_host, vel_local_host))
        
    except Exception as e:
        print(f"ERRO CRÍTICO NA GPU {gpu_id}: {str(e)}")
    finally:
        ctx.pop()

def simular_n_corpos_multigpu(pos, vel, massas, passos, dt, G, eps, num_gpus_ativas=4):
    N = pos.shape[0]
    
    # 1. Preparar a Memória Partilhada (Shared Memory Raw Array - evita Lock/Pickle overhead)
    pos_mass = np.zeros((N, 4), dtype=np.float32)
    pos_mass[:, :3] = pos
    pos_mass[:, 3] = massas
    
    shared_pos_mass_base = mp.RawArray('f', N * 4)
    shared_pos_mass_np = np.frombuffer(shared_pos_mass_base, dtype=np.float32).reshape((N, 4))
    np.copyto(shared_pos_mass_np, pos_mass)
    
    barrier_init = mp.Barrier(num_gpus_ativas + 1) # +1 para o CPU (Host) esperar também
    barrier_step = mp.Barrier(num_gpus_ativas)     # Apenas para as GPUs se sincronizarem
    queue_result = mp.Queue()
    processos = []
    
    # 2. Despachar Trabalho
    for i in range(num_gpus_ativas):
        N_local_base = N // num_gpus_ativas
        offset = i * N_local_base
        N_local = N_local_base if i < num_gpus_ativas - 1 else N - offset
        
        vel_local_init = vel[offset:offset+N_local].copy()
        
        p = mp.Process(target=worker_gpu, args=(
            i, num_gpus_ativas, N, vel_local_init, passos, dt, G, eps, 
            shared_pos_mass_base, barrier_init, barrier_step, queue_result
        ))
        p.start()
        processos.append(p)
        
    # O Host espera que as GPUs compilem o código e aloquem memória
    barrier_init.wait()
    
    # Iniciar o cronómetro APENAS para a simulação real
    inicio_tempo = time.perf_counter()
    
    # Resgatar os resultados ANTES do join para evitar Deadlocks de memória
    resultados = []
    for _ in range(num_gpus_ativas):
        resultados.append(queue_result.get())
        
    for p in processos:
        p.join()
        
    fim_tempo = time.perf_counter()
    
    # Reconstruir os arrays finais
    resultados.sort(key=lambda x: x[0])
    pos_final = np.zeros((N, 3), dtype=np.float32)
    vel_final = np.zeros((N, 3), dtype=np.float32)
    
    for res in resultados:
        gpu_id, pos_local, vel_local = res
        N_local_base = N // num_gpus_ativas
        offset = gpu_id * N_local_base
        N_local = len(pos_local)
        
        pos_final[offset:offset+N_local] = pos_local[:, :3]
        vel_final[offset:offset+N_local] = vel_local.reshape((N_local, 3))
        
    return pos_final, vel_final, (fim_tempo - inicio_tempo)
