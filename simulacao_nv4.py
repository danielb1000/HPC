import threading
import queue
import numpy as np
import time

# Reutilizamos o mesmo kernel, a otimização está na forma como o Python orquestra a memória
KERNEL_CODE_NV4 = """
#define BLOCK_SIZE 1024

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
    int global_i = offset + i; 

    if (i < N_local) {
        float4 my_pos = pos_mass_local[i];
        pos_i_x = my_pos.x; pos_i_y = my_pos.y; pos_i_z = my_pos.z;
    }

    int num_tiles = (N_total + blockDim.x - 1) / blockDim.x;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int j_shared_load_idx = tile_idx * blockDim.x + threadIdx.x;
        
        if (j_shared_load_idx < N_total) {
            s_pos_mass[threadIdx.x] = pos_mass_all[j_shared_load_idx];
        } else {
            s_pos_mass[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();

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
        acel_local[i * 3 + 0] = ax; acel_local[i * 3 + 1] = ay; acel_local[i * 3 + 2] = az;
    }
}

__global__ void post_update_multigpu(float *vel_local, float *acel_local, float dt, int N_local) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N_local) {
        for (int d = 0; d < 3; d++) {
            vel_local[i * 3 + d] += 0.5f * acel_local[i * 3 + d] * dt; 
        }
    }
}
"""

def worker_gpu_nvlink(gpu_id, num_gpus, N_total, vel_init_local, pos_init_all, passos, dt, G, eps, 
                      pointers_dict, contexts_dict, barrier_init, barrier_step, queue_result):
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    
    cuda.init()
    dev = cuda.Device(gpu_id)
    ctx = dev.make_context()
    contexts_dict[gpu_id] = ctx
    
    try:
        mod = SourceModule(KERNEL_CODE_NV4, options=["-Xptxas", "-v", "-O3"])
        kernel_acel = mod.get_function("calcular_aceleracoes_multigpu")
        kernel_pre = mod.get_function("pre_update_multigpu")
        kernel_post = mod.get_function("post_update_multigpu")
        
        N_local_base = N_total // num_gpus
        offset = gpu_id * N_local_base
        N_local = N_local_base if gpu_id < num_gpus - 1 else N_total - offset
        
        # Alocar as posições do universo completo nesta GPU
        d_pos_mass_all = cuda.mem_alloc(pos_init_all.nbytes)
        cuda.memcpy_htod(d_pos_mass_all, pos_init_all)
        
        # Criar um buffer exclusivo da nossa fatia. 
        # Esta é a memória que vamos exportar para ser lida via NVLink pelas outras GPUs.
        tamanho_fatia_bytes = N_local * 16 # 4 floats (float4) de 4 bytes = 16 bytes por partícula
        d_pos_mass_local = cuda.mem_alloc(tamanho_fatia_bytes)
        cuda.memcpy_dtod(d_pos_mass_local, int(d_pos_mass_all) + (offset * 16), tamanho_fatia_bytes)
        
        vel_local_host = vel_init_local.flatten()
        d_vel_local = cuda.mem_alloc(vel_local_host.nbytes)
        d_acel_local = cuda.mem_alloc(N_local * 3 * 4)
        cuda.memcpy_htod(d_vel_local, vel_local_host)

        # Partilhar o ponteiro direto da VRAM (possível porque partilhamos o mesmo processo)
        pointers_dict[gpu_id] = (int(d_pos_mass_local), offset, N_local)

        # Sincronizar: Dar tempo para todas as GPUs publicarem os seus ponteiros
        barrier_init.wait()

        # Mapear os ponteiros das outras GPUs diretamente para a nossa VRAM local
        peer_buffers = {}
        for peer_id in range(num_gpus):
            if peer_id != gpu_id:
                peer_ptr, peer_offset, peer_N = pointers_dict[peer_id]
                peer_ctx = contexts_dict[peer_id]
                if dev.can_access_peer(cuda.Device(peer_id)):
                    try:
                        ctx.enable_peer_access(peer_ctx)
                    except Exception:
                        pass # Ignorar se o NVLink já estiver ativado
                    peer_buffers[peer_id] = (peer_ptr, peer_offset, peer_N)
                else:
                    print(f"Aviso GPU {gpu_id}: NVLink P2P não suportado para a GPU {peer_id}.")

        block_dim = (1024, 1, 1)
        grid_dim = (int(np.ceil(N_local / 1024)), 1)
        
        kernel_acel(d_pos_mass_all, d_pos_mass_local, d_acel_local, np.int32(N_total), np.int32(N_local), np.int32(offset), np.float32(G), np.float32(eps), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()
        barrier_init.wait()
        
        for passo in range(passos):
            kernel_pre(d_pos_mass_local, d_vel_local, d_acel_local, np.float32(dt), np.int32(N_local), block=block_dim, grid=grid_dim)
            cuda.memcpy_dtod(int(d_pos_mass_all) + (offset * 16), d_pos_mass_local, N_local * 16)
            
            barrier_step.wait() # Esperar pelo fim dos cálculos físicos locais
            
            # O MAGIA ACONTECE AQUI: Em vez de copiar para a CPU, pedimos às outras 
            # GPUs para nos passarem os dados diretamente pela bridge NVLink (Device-to-Device).
            for peer_id, (peer_ptr, peer_offset, peer_N) in peer_buffers.items():
                dest_ptr = int(d_pos_mass_all) + (peer_offset * 16)
                cuda.memcpy_dtod(dest_ptr, peer_ptr, peer_N * 16)
                
            cuda.Context.synchronize()
            
            barrier_step.wait() # Aguardar que todos terminem as leituras P2P antes de avançar e modificar/libertar a memória
            
            kernel_acel(d_pos_mass_all, d_pos_mass_local, d_acel_local, np.int32(N_total), np.int32(N_local), np.int32(offset), np.float32(G), np.float32(eps), block=block_dim, grid=grid_dim)
            kernel_post(d_vel_local, d_acel_local, np.float32(dt), np.int32(N_local), block=block_dim, grid=grid_dim)

        pos_mass_local_final = np.empty((N_local, 4), dtype=np.float32)
        vel_local_final = np.empty((N_local, 3), dtype=np.float32)
        cuda.memcpy_dtoh(pos_mass_local_final, d_pos_mass_local)
        cuda.memcpy_dtoh(vel_local_final, d_vel_local)
        
        queue_result.put((gpu_id, pos_mass_local_final, vel_local_final))
        
    finally:
        ctx.pop()

def simular_n_corpos_nv4(pos, vel, massas, passos, dt, G, eps, num_gpus_ativas=4):
    N = pos.shape[0]
    pos_mass = np.zeros((N, 4), dtype=np.float32)
    pos_mass[:, :3] = pos
    pos_mass[:, 3] = massas
    
    pointers_dict = {}
    contexts_dict = {}
    barrier_init = threading.Barrier(num_gpus_ativas + 1) 
    barrier_step = threading.Barrier(num_gpus_ativas)     
    queue_result = queue.Queue()
    processos = []
    
    for i in range(num_gpus_ativas):
        N_local_base = N // num_gpus_ativas
        offset = i * N_local_base
        N_local = N_local_base if i < num_gpus_ativas - 1 else N - offset
        vel_local_init = vel[offset:offset+N_local].copy()
        
        p = threading.Thread(target=worker_gpu_nvlink, args=(
            i, num_gpus_ativas, N, vel_local_init, pos_mass, passos, dt, G, eps, 
            pointers_dict, contexts_dict, barrier_init, barrier_step, queue_result
        ))
        p.start()
        processos.append(p)
        
    barrier_init.wait() # Fase 1: Sincronizar após emissão de IPC
    
    inicio_tempo = time.perf_counter()
    barrier_init.wait() # Início real da simulação 
    
    resultados = []
    for _ in range(num_gpus_ativas):
        resultados.append(queue_result.get())
        
    for p in processos:
        p.join()
        
    fim_tempo = time.perf_counter()
    
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