import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
import time

# Falta comentar o codigo do GPU para saber onde posso melhorar a performance, e onde posso melhorar a legibilidade do codigo.
kernel_code = """
__global__ void pre_update(float *pos, float *vel, float *acel, float dt, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        for (int d = 0; d < 3; d++) {
            int idx = i * 3 + d;
            float v_half = vel[idx] + 0.5f * acel[idx] * dt;
            pos[idx] += v_half * dt;
            vel[idx] = v_half; 
        }
    }
}

__global__ void calcular_aceleracoes(float *pos, float *massas, float *acel, int N, float G, float eps) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        float ax = 0.0f; float ay = 0.0f; float az = 0.0f;
        float pos_i_x = pos[i * 3 + 0];
        float pos_i_y = pos[i * 3 + 1];
        float pos_i_z = pos[i * 3 + 2];

        for (int j = 0; j < N; j++) {
            if (i != j) {
                float dx = pos[j * 3 + 0] - pos_i_x;
                float dy = pos[j * 3 + 1] - pos_i_y;
                float dz = pos[j * 3 + 2] - pos_i_z;
                float dist_sq = dx*dx + dy*dy + dz*dz;
                float denominador = powf(dist_sq + eps*eps, 1.5f);
                float forca = (G * massas[j]) / denominador;
                ax += forca * dx; ay += forca * dy; az += forca * dz;
            }
        }
        acel[i * 3 + 0] = ax; acel[i * 3 + 1] = ay; acel[i * 3 + 2] = az;
    }
}

__global__ void post_update(float *vel, float *acel, float dt, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        for (int d = 0; d < 3; d++) {
            int idx = i * 3 + d;
            vel[idx] += 0.5f * acel[idx] * dt; 
        }
    }
}
"""

mod = SourceModule(kernel_code)
pre_update_gpu = mod.get_function("pre_update")
calcular_aceleracoes_gpu = mod.get_function("calcular_aceleracoes")
post_update_gpu = mod.get_function("post_update")

def simular_n_corpos_gpu(pos, vel, massas, passos, dt, G, eps):
    N = pos.shape[0]
    N_numpy = np.int32(N)
    
    # Achatar arrays de (N, 3) para 1D (N*3) para a memória do C++
    pos_flat = pos.flatten().astype(np.float32)
    vel_flat = vel.flatten().astype(np.float32)
    acel_flat = np.zeros(N * 3, dtype=np.float32)
    massas = massas.astype(np.float32)

    pos_gpu = cuda.mem_alloc(pos_flat.nbytes)
    vel_gpu = cuda.mem_alloc(vel_flat.nbytes)
    massas_gpu = cuda.mem_alloc(massas.nbytes)
    acel_gpu = cuda.mem_alloc(acel_flat.nbytes)

    cuda.memcpy_htod(pos_gpu, pos_flat)
    cuda.memcpy_htod(vel_gpu, vel_flat)
    cuda.memcpy_htod(massas_gpu, massas)
    cuda.memcpy_htod(acel_gpu, acel_flat)

    threads_por_bloco = 256
    blocos_por_grid = int(np.ceil(N / threads_por_bloco))
    block_dim = (threads_por_bloco, 1, 1)
    grid_dim = (blocos_por_grid, 1)

    # Aceleração inicial (t=0)
    calcular_aceleracoes_gpu(pos_gpu, massas_gpu, acel_gpu, N_numpy, np.float32(G), np.float32(eps), block=block_dim, grid=grid_dim)

    # Sincronizar e iniciar cronómetro só para o ciclo
    cuda.Context.synchronize()
    inicio_tempo = time.perf_counter()

    for _ in range(passos):
        pre_update_gpu(pos_gpu, vel_gpu, acel_gpu, np.float32(dt), N_numpy, block=block_dim, grid=grid_dim)
        calcular_aceleracoes_gpu(pos_gpu, massas_gpu, acel_gpu, N_numpy, np.float32(G), np.float32(eps), block=block_dim, grid=grid_dim)
        post_update_gpu(vel_gpu, acel_gpu, np.float32(dt), N_numpy, block=block_dim, grid=grid_dim)

    cuda.Context.synchronize()
    fim_tempo = time.perf_counter()

    # Recuperar dados
    cuda.memcpy_dtoh(pos_flat, pos_gpu)
    
    # Reconstruir formato (N, 3)
    pos_final = pos_flat.reshape((N, 3))

    return pos_final, (fim_tempo - inicio_tempo)