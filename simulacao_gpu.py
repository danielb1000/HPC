import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
import numpy as np

THREADS_POR_BLOCO = 256 

kernel_code = """
#define BLOCK_SIZE %d

__global__ void pre_update(float *pos, float *vel, float *acel, float dt, int N) {
    // Este kernel implementa a primeira metade do integrador Leapfrog (Kick-Drift-Kick).
    // 1. Calcula a velocidade no meio do passo de tempo (v_half).
    // 2. Atualiza a posição para o fim do passo de tempo (Drift).
    // 3. Armazena a v_half de volta no array de velocidade para ser usada no post_update.
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        // Calcular primeiro todas as componentes da velocidade de meio-passo
        float v_half_x = vel[i*3 + 0] + 0.5f * acel[i*3 + 0] * dt;
        float v_half_y = vel[i*3 + 1] + 0.5f * acel[i*3 + 1] * dt;
        float v_half_z = vel[i*3 + 2] + 0.5f * acel[i*3 + 2] * dt;
        // Atualizar a posição usando a velocidade de meio-passo
        pos[i*3 + 0] += v_half_x * dt;
        pos[i*3 + 1] += v_half_y * dt;
        pos[i*3 + 2] += v_half_z * dt;
        // Guardar a velocidade de meio-passo para o kernel post_update
        vel[i*3 + 0] = v_half_x;
        vel[i*3 + 1] = v_half_y;
        vel[i*3 + 2] = v_half_z;
    }
}

__global__ void pre_update_float4(float4 *pos_mass, float *vel, float *acel, float dt, int N) {
    // Versão do pre_update adaptada para o array empacotado float4.
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        float4 p = pos_mass[i]; // Lê pos(X,Y,Z) e a massa(W) de uma vez
        
        float v_half_x = vel[i*3 + 0] + 0.5f * acel[i*3 + 0] * dt;
        float v_half_y = vel[i*3 + 1] + 0.5f * acel[i*3 + 1] * dt;
        float v_half_z = vel[i*3 + 2] + 0.5f * acel[i*3 + 2] * dt;

        p.x += v_half_x * dt;
        p.y += v_half_y * dt;
        p.z += v_half_z * dt;

        pos_mass[i] = p; // Guarda a nova posição e a massa de volta na memória global
        
        vel[i*3 + 0] = v_half_x; vel[i*3 + 1] = v_half_y; vel[i*3 + 2] = v_half_z;
    }
}

/*
Kernel ingénuo (naive) para o cálculo de acelerações.
Serve como baseline de performance. A sua principal limitação é o acesso
intensivo e repetido à memória global, que é lenta.
*/
__global__ void calcular_aceleracoes_naive(float *pos, float *massas, float *acel, int N, float G, float eps) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        // Otimização Possível: Carregar pos_i para registos é bom.
        float pos_i_x = pos[i * 3 + 0];
        float pos_i_y = pos[i * 3 + 1];
        float pos_i_z = pos[i * 3 + 2];

        // Ponto Crítico de Performance: Este ciclo é o coração do O(N^2).
        // Para cada partícula 'i', estamos a ler as posições de TODAS as outras 'j'
        // diretamente da memória global da GPU, que é lenta.
        for (int j = 0; j < N; j++) {
            if (i != j) {
                // Cada leitura de pos[j*3+d] é um acesso à memória global.
                float dx = pos[j * 3 + 0] - pos_i_x;
                float dy = pos[j * 3 + 1] - pos_i_y;
                float dz = pos[j * 3 + 2] - pos_i_z;
                float dist_sq = dx*dx + dy*dy + dz*dz;

                // Naive implementation: usa powf padrão de alta precisão, mas lento.
                float denominador = powf(dist_sq + eps*eps, 1.5f);
                float forca = (G * massas[j]) / denominador;
                ax += forca * dx; ay += forca * dy; az += forca * dz;
            }
        }
        acel[i * 3 + 0] = ax; acel[i * 3 + 1] = ay; acel[i * 3 + 2] = az;
    }
}

/*
Kernel Naive com otimização Fast Math (rsqrtf).
Isola o ganho de performance puramente matemático antes de introduzir otimizações de memória.
*/
__global__ void calcular_aceleracoes_naive_fast_math(float *pos, float *massas, float *acel, int N, float G, float eps) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        float pos_i_x = pos[i * 3 + 0];
        float pos_i_y = pos[i * 3 + 1];
        float pos_i_z = pos[i * 3 + 2];

        for (int j = 0; j < N; j++) {
            if (i != j) {
                float dx = pos[j * 3 + 0] - pos_i_x;
                float dy = pos[j * 3 + 1] - pos_i_y;
                float dz = pos[j * 3 + 2] - pos_i_z;
                float dist_sq = dx*dx + dy*dy + dz*dz;

                float inv_dist = rsqrtf(dist_sq + eps*eps);
                float forca = G * massas[j] * inv_dist * inv_dist * inv_dist;
                ax += forca * dx; ay += forca * dy; az += forca * dz;
            }
        }
        acel[i * 3 + 0] = ax; acel[i * 3 + 1] = ay; acel[i * 3 + 2] = az;
    }
}

/*
Kernel otimizado (shared_mem) para calcular acelerações. Utiliza uma técnica principal:
1. Memória Partilhada (Shared Memory): Reduz drasticamente os acessos à lenta memória global.
   Cada bloco de threads carrega um "tile" (pedaço) de partículas para a memória partilhada,
   que é ordens de magnitude mais rápida.
*/
__global__ void calcular_aceleracoes_shared_mem(float *pos, float *massas, float *acel, int N, float G, float eps) {
    // Memória partilhada para um bloco de partículas. O tamanho é fixo e deve
    // corresponder ao `threads_por_bloco` (e.g., 256).
    __shared__ float s_pos[BLOCK_SIZE * 3];
    __shared__ float s_massas[BLOCK_SIZE];

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Variáveis locais para as partículas válidas acumularem as forças
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float pos_i_x = 0.0f, pos_i_y = 0.0f, pos_i_z = 0.0f;

    // Carregar a posição da "minha" partícula para registos privados (apenas se for válida)
    if (i < N) {
        pos_i_x = pos[i * 3 + 0];
        pos_i_y = pos[i * 3 + 1];
        pos_i_z = pos[i * 3 + 2];
    }

    int num_tiles = (N + blockDim.x - 1) / blockDim.x;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int j_shared_load_idx = tile_idx * blockDim.x + threadIdx.x;
        
        // TODAS as threads (mesmo as i >= N) participam no carregamento para a memória partilhada!
        if (j_shared_load_idx < N) {
            s_pos[threadIdx.x * 3 + 0] = pos[j_shared_load_idx * 3 + 0];
            s_pos[threadIdx.x * 3 + 1] = pos[j_shared_load_idx * 3 + 1];
            s_pos[threadIdx.x * 3 + 2] = pos[j_shared_load_idx * 3 + 2];
            s_massas[threadIdx.x] = massas[j_shared_load_idx];
        } else {
            s_massas[threadIdx.x] = 0.0f;
            s_pos[threadIdx.x * 3 + 0] = 0.0f;
            s_pos[threadIdx.x * 3 + 1] = 0.0f;
            s_pos[threadIdx.x * 3 + 2] = 0.0f;
        }
        __syncthreads(); // BARREIRA SEGURA: Nenhuma thread faltou ao encontro!

        // Apenas as partículas válidas fazem o trabalho pesado de O(N^2)
        if (i < N) {
            #pragma unroll
            for (int j_tile = 0; j_tile < blockDim.x; j_tile++) {
                if (s_massas[j_tile] > 0.0f && (tile_idx * blockDim.x + j_tile) != i) {
                    float dx = s_pos[j_tile * 3 + 0] - pos_i_x;
                    float dy = s_pos[j_tile * 3 + 1] - pos_i_y;
                    float dz = s_pos[j_tile * 3 + 2] - pos_i_z;
                    float dist_sq = dx*dx + dy*dy + dz*dz;
                    float inv_dist = rsqrtf(dist_sq + eps*eps);
                    float forca = G * s_massas[j_tile] * inv_dist * inv_dist * inv_dist;
                    ax += forca * dx; ay += forca * dy; az += forca * dz;
                }
            }
        }
        __syncthreads(); // BARREIRA SEGURA antes do próximo tile
    }

    // Gravar resultados na memória global (apenas partículas válidas)
    if (i < N) {
        acel[i * 3 + 0] = ax; acel[i * 3 + 1] = ay; acel[i * 3 + 2] = az;
    }
}

/*
Kernel OTIMIZAÇÃO MÁXIMA (shared_mem_float4).
Funde as Posições e as Massas num array único do tipo `float4` (X, Y, Z, W=Massa).
Isto permite que a GPU leia 128-bits de dados contíguos de uma só vez por thread,
maximizando a largura de banda (Memory Coalescing perfeito).
Mantém a mesma arquitetura robusta de Tiling e syncthreads que estabilizámos.
*/
__global__ void calcular_aceleracoes_shared_mem_float4(float4 *pos_mass, float *acel, int N, float G, float eps) {
    __shared__ float4 s_pos_mass[BLOCK_SIZE]; // Um único array partilhado empacotado!

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float pos_i_x = 0.0f, pos_i_y = 0.0f, pos_i_z = 0.0f;

    if (i < N) {
        float4 my_pos_mass = pos_mass[i]; // 1 Leitura massiva de 128-bits
        pos_i_x = my_pos_mass.x; pos_i_y = my_pos_mass.y; pos_i_z = my_pos_mass.z;
    }

    int num_tiles = (N + blockDim.x - 1) / blockDim.x;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int j_shared_load_idx = tile_idx * blockDim.x + threadIdx.x;
        
        if (j_shared_load_idx < N) {
            s_pos_mass[threadIdx.x] = pos_mass[j_shared_load_idx];
        } else {
            s_pos_mass[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f); // massa fantasma (w=0)
        }
        __syncthreads();

        if (i < N) {
            #pragma unroll
            for (int j_tile = 0; j_tile < blockDim.x; j_tile++) {
                float4 p_j = s_pos_mass[j_tile];
                if (p_j.w > 0.0f && (tile_idx * blockDim.x + j_tile) != i) {
                    float dx = p_j.x - pos_i_x; float dy = p_j.y - pos_i_y; float dz = p_j.z - pos_i_z;
                    float dist_sq = dx*dx + dy*dy + dz*dz;
                    float inv_dist = rsqrtf(dist_sq + eps*eps);
                    float forca = G * p_j.w * inv_dist * inv_dist * inv_dist; // p_j.w é a massa!
                    ax += forca * dx; ay += forca * dy; az += forca * dz;
                }
            }
        }
        __syncthreads();
    }
    if (i < N) {
        acel[i * 3 + 0] = ax; acel[i * 3 + 1] = ay; acel[i * 3 + 2] = az;
    }
}

__global__ void post_update(float *vel, float *acel, float dt, int N) {
    // Este kernel implementa a segunda metade do integrador Leapfrog.
    // Ele recebe a aceleração recém-calculada (acel) e a velocidade de meio-passo
    // (armazenada em vel) para calcular a velocidade final do passo de tempo.
    // v(t+dt) = v(t+dt/2) + a(t+dt) * dt/2
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        for (int d = 0; d < 3; d++) {
            int idx = i * 3 + d;
            vel[idx] += 0.5f * acel[idx] * dt; 
        }
    }
}

__global__ void calcular_energia(float *pos, float *vel, float *massas, float *energia, int N, float G, float eps) {
    // Kernel de diagnóstico para calcular a Energia Total (Cinética + Potencial) na GPU.
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        // 1. Energia Cinética: 1/2 * m * v^2
        float v_sq = vel[i*3 + 0]*vel[i*3 + 0] + vel[i*3 + 1]*vel[i*3 + 1] + vel[i*3 + 2]*vel[i*3 + 2];
        float e_kin = 0.5f * massas[i] * v_sq;

        // 2. Energia Potencial: soma( -G * m_i * m_j / r )
        float e_pot = 0.0f;
        float pos_i_x = pos[i*3 + 0];
        float pos_i_y = pos[i*3 + 1];
        float pos_i_z = pos[i*3 + 2];
        float m_i = massas[i];

        for (int j = 0; j < N; j++) {
            if (i != j) {
                float dx = pos[j*3 + 0] - pos_i_x;
                float dy = pos[j*3 + 1] - pos_i_y;
                float dz = pos[j*3 + 2] - pos_i_z;
                float dist_sq = dx*dx + dy*dy + dz*dz;
                
                // rsqrtf calcula 1/sqrt(dist_sq), o que é exatamente 1/r
                float inv_dist = rsqrtf(dist_sq + eps*eps); 
                e_pot += -G * m_i * massas[j] * inv_dist;
            }
        }
        // Como somamos todos os pares (i,j) e (j,i), a energia potencial está duplicada.
        // Dividimos por 2 para obter o valor real.
        e_pot *= 0.5f;

        energia[i] = e_kin + e_pot;
    }
}

""" % THREADS_POR_BLOCO

mod = None
pre_update_gpu = None
pre_update_float4_gpu = None
post_update_gpu = None
kernels_aceleracao = {}

def _compilar_kernels():
    """Compila os kernels apenas quando necessário (Lazy Compilation). Evita crash de contexto ao ser importado."""
    global mod, pre_update_gpu, pre_update_float4_gpu, post_update_gpu, kernels_aceleracao
    if mod is None:
        mod = SourceModule(kernel_code, options=["-Xptxas", "-v", "-O3"])
        pre_update_gpu = mod.get_function("pre_update")
        pre_update_float4_gpu = mod.get_function("pre_update_float4")
        post_update_gpu = mod.get_function("post_update")
        
        kernels_aceleracao = {
            "naive": mod.get_function("calcular_aceleracoes_naive"),
            "naive_fast_math": mod.get_function("calcular_aceleracoes_naive_fast_math"),
            "shared_mem": mod.get_function("calcular_aceleracoes_shared_mem"),
            "shared_mem_float4": mod.get_function("calcular_aceleracoes_shared_mem_float4")
        }

def validar_energia_gpu(pos, vel, massas, G, eps):
    """
    Executa o cálculo pesado de energia O(N^2) na GPU e devolve a Energia Total.
    """
    _compilar_kernels()
    N = pos.shape[0]
    
    pos_flat = pos.flatten().astype(np.float32)
    vel_flat = vel.flatten().astype(np.float32)
    massas_flat = massas.astype(np.float32)
    energia_flat = np.zeros(N, dtype=np.float32)
    
    pos_gpu = cuda.mem_alloc(pos_flat.nbytes)
    vel_gpu = cuda.mem_alloc(vel_flat.nbytes)
    massas_gpu = cuda.mem_alloc(massas_flat.nbytes)
    energia_gpu = cuda.mem_alloc(energia_flat.nbytes)
    
    cuda.memcpy_htod(pos_gpu, pos_flat)
    cuda.memcpy_htod(vel_gpu, vel_flat)
    cuda.memcpy_htod(massas_gpu, massas_flat)
    
    threads_por_bloco = THREADS_POR_BLOCO
    blocos_por_grid = int(np.ceil(N / threads_por_bloco))
    
    calcular_energia_kernel = mod.get_function("calcular_energia")
    calcular_energia_kernel(pos_gpu, vel_gpu, massas_gpu, energia_gpu, np.int32(N), np.float32(G), np.float32(eps), block=(threads_por_bloco, 1, 1), grid=(blocos_por_grid, 1))
    
    cuda.memcpy_dtoh(energia_flat, energia_gpu)
    
    return np.sum(energia_flat) # O CPU faz apenas a soma final O(N)

def simular_n_corpos_gpu(pos, vel, massas, passos, dt, G, eps, method: str = "naive"):
    _compilar_kernels()
    N = pos.shape[0]

    N_numpy = np.int32(N)
    
    # Preparar e alocar memória na GPU
    vel_flat = vel.flatten().astype(np.float32)
    acel_flat = np.zeros(N * 3, dtype=np.float32)
    
    vel_gpu = cuda.mem_alloc(vel_flat.nbytes)
    acel_gpu = cuda.mem_alloc(acel_flat.nbytes)
    
    cuda.memcpy_htod(vel_gpu, vel_flat)
    cuda.memcpy_htod(acel_gpu, acel_flat)
    
    is_float4 = (method == "shared_mem_float4")

    if is_float4:
        # Empacotar Posições (X,Y,Z) e Massas (W) no mesmo array NumPy de (N, 4)
        pos_mass = np.zeros((N, 4), dtype=np.float32)
        pos_mass[:, :3] = pos
        pos_mass[:, 3] = massas
        pos_flat = pos_mass.flatten()
        pos_gpu = cuda.mem_alloc(pos_flat.nbytes)
        cuda.memcpy_htod(pos_gpu, pos_flat)
    else:
        pos_flat = pos.flatten().astype(np.float32)
        massas_flat = massas.astype(np.float32)
        pos_gpu = cuda.mem_alloc(pos_flat.nbytes)
        massas_gpu = cuda.mem_alloc(massas_flat.nbytes)
        cuda.memcpy_htod(pos_gpu, pos_flat)
        cuda.memcpy_htod(massas_gpu, massas_flat)

    threads_por_bloco = THREADS_POR_BLOCO
    blocos_por_grid_normal = int(np.ceil(N / threads_por_bloco))
        
    block_dim = (threads_por_bloco, 1, 1)
    grid_dim_normal = (blocos_por_grid_normal, 1)

    if method not in kernels_aceleracao:
        raise ValueError(f"Método de kernel '{method}' desconhecido. Válidos: {list(kernels_aceleracao.keys())}")
    
    kernel_acel = kernels_aceleracao[method]

    # Aceleração inicial (t=0)
    if is_float4:
        kernel_acel(pos_gpu, acel_gpu, N_numpy, np.float32(G), np.float32(eps), block=block_dim, grid=grid_dim_normal)
    else:
        kernel_acel(pos_gpu, massas_gpu, acel_gpu, N_numpy, np.float32(G), np.float32(eps), block=block_dim, grid=grid_dim_normal)

    # Sincronizar e iniciar cronómetro só para o ciclo
    cuda.Context.synchronize()
    inicio_tempo = time.perf_counter()

    for _ in range(passos):
        if is_float4:
            pre_update_float4_gpu(pos_gpu, vel_gpu, acel_gpu, np.float32(dt), N_numpy, block=block_dim, grid=grid_dim_normal)
            kernel_acel(pos_gpu, acel_gpu, N_numpy, np.float32(G), np.float32(eps), block=block_dim, grid=grid_dim_normal)
        else:
            pre_update_gpu(pos_gpu, vel_gpu, acel_gpu, np.float32(dt), N_numpy, block=block_dim, grid=grid_dim_normal)
            kernel_acel(pos_gpu, massas_gpu, acel_gpu, N_numpy, np.float32(G), np.float32(eps), block=block_dim, grid=grid_dim_normal)
            
        post_update_gpu(vel_gpu, acel_gpu, np.float32(dt), N_numpy, block=block_dim, grid=grid_dim_normal)

    cuda.Context.synchronize()
    fim_tempo = time.perf_counter()

    # Recuperar dados e formatar
    cuda.memcpy_dtoh(pos_flat, pos_gpu)
    cuda.memcpy_dtoh(vel_flat, vel_gpu)
    
    if is_float4:
        pos_final = pos_flat.reshape((N, 4))[:, :3]
    else:
        pos_final = pos_flat.reshape((N, 3))

    vel_final = vel_flat.reshape((N, 3))

    return pos_final, vel_final, (fim_tempo - inicio_tempo)