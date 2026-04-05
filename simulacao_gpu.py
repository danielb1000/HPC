import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time

# Falta comentar o codigo do GPU para saber onde posso melhorar a performance, e onde posso melhorar a legibilidade do codigo.
kernel_code = """
__global__ void pre_update(float *pos, float *vel, float *acel, float dt, int N) {
    // Este kernel implementa a primeira metade do integrador Leapfrog (Kick-Drift-Kick).
    // 1. Calcula a velocidade no meio do passo de tempo (v_half).
    // 2. Atualiza a posição para o fim do passo de tempo (Drift).
    // 3. Armazena a v_half de volta no array de velocidade para ser usada no post_update.
    // Otimização possível: O loop 'for (int d=...)' pode ser desenrolado para evitar
    // ramificações, embora o compilador NVCC geralmente faça isso automaticamente.
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

                // Otimização Possível: A função powf é computacionalmente cara.
                // Para expoentes como 1.5, é muito mais rápido usar a instrução
                // intrínseca rsqrtf (recíproca da raiz quadrada).
                float denominador = powf(dist_sq + eps*eps, 1.5f);
                float forca = (G * massas[j]) / denominador;
                ax += forca * dx; ay += forca * dy; az += forca * dz;
            }
        }
        acel[i * 3 + 0] = ax; acel[i * 3 + 1] = ay; acel[i * 3 + 2] = az;
    }
}

/*
Kernel otimizado (shared_mem) para calcular acelerações. Utiliza duas técnicas principais:
1. Memória Partilhada (Shared Memory): Reduz drasticamente os acessos à lenta memória global.
   Cada bloco de threads carrega um "tile" (pedaço) de partículas para a memória partilhada,
   que é ordens de magnitude mais rápida.
2. Otimização de Instruções: Substitui a função `powf` pela intrínseca `rsqrtf`,
   que é muito mais rápida no hardware da GPU. 1 / (d^3) é calculado como (1/sqrt(d^2))^3.
*/
__global__ void calcular_aceleracoes_shared_mem(float *pos, float *massas, float *acel, int N, float G, float eps) {
    // Memória partilhada para um bloco de partículas. O tamanho é fixo e deve
    // corresponder ao `threads_por_bloco` (e.g., 256).
    __shared__ float s_pos[256 * 3];
    __shared__ float s_massas[256];

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Apenas threads que correspondem a partículas válidas (i < N) devem trabalhar.
    if (i < N) {
        // Carregar a posição da "minha" partícula para registos privados.
        float pos_i_x = pos[i * 3 + 0];
        float pos_i_y = pos[i * 3 + 1];
        float pos_i_z = pos[i * 3 + 2];
        float ax = 0.0f, ay = 0.0f, az = 0.0f;

        int num_tiles = (N + blockDim.x - 1) / blockDim.x;

        for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
            int j_shared_load_idx = tile_idx * blockDim.x + threadIdx.x;
            // Carregar dados para a memória partilhada.
            // A correção crítica está aqui: se uma thread não corresponde a uma partícula
            // válida (porque j_shared_load_idx >= N), ela DEVE carregar uma massa de 0.0f.
            // O código anterior não fazia isso, deixando "lixo" de iterações passadas na
            // memória partilhada, o que causava a instabilidade numérica.
            if (j_shared_load_idx < N) {
                s_pos[threadIdx.x * 3 + 0] = pos[j_shared_load_idx * 3 + 0];
                s_pos[threadIdx.x * 3 + 1] = pos[j_shared_load_idx * 3 + 1];
                s_pos[threadIdx.x * 3 + 2] = pos[j_shared_load_idx * 3 + 2];
                s_massas[threadIdx.x] = massas[j_shared_load_idx];
            } else {
                s_massas[threadIdx.x] = 0.0f; // Partícula fantasma, força será zero.
            }
            __syncthreads();

            // Como agora garantimos que as partículas "fantasma" têm massa zero,
            // podemos iterar sobre todo o bloco de threads (blockDim.x) sem perigo.
            // A verificação s_massas[j_tile] > 0.0f garante que não fazemos cálculos
            // desnecessários e também que o índice global da partícula j é válido.
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
            __syncthreads();
        }
        acel[i * 3 + 0] = ax; acel[i * 3 + 1] = ay; acel[i * 3 + 2] = az;
    }
}

/*
Kernel OTIMIZAÇÃO MÁXIMA (shared_mem_float4).
Além da memória partilhada e rsqrtf, funde as Posições e as Massas num array
único do tipo `float4` (X, Y, Z, W=Massa).
Isto permite que a GPU leia 128-bits de dados contíguos de uma só vez por thread,
maximizando a largura de banda (Memory Coalescing perfeito).
*/
__global__ void calcular_aceleracoes_shared_mem_float4(float4 *pos_mass, float *acel, int N, float G, float eps) {
    __shared__ float4 s_pos_mass[256]; // Array partilhado empacotado!

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N) {
        float4 my_pos_mass = pos_mass[i]; // 1 Leitura vetorizada massiva em vez de 4 dispersas!
        float pos_i_x = my_pos_mass.x;
        float pos_i_y = my_pos_mass.y;
        float pos_i_z = my_pos_mass.z;
        float ax = 0.0f, ay = 0.0f, az = 0.0f;

        int num_tiles = (N + blockDim.x - 1) / blockDim.x;

        for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
            int j_shared_load_idx = tile_idx * blockDim.x + threadIdx.x;
            
            if (j_shared_load_idx < N) {
                s_pos_mass[threadIdx.x] = pos_mass[j_shared_load_idx]; // Leitura global ultra-rápida
            } else {
                s_pos_mass[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f); // massa fantasma
            }
            __syncthreads();

            #pragma unroll
            for (int j_tile = 0; j_tile < blockDim.x; j_tile++) {
                float4 p_j = s_pos_mass[j_tile];
                if (p_j.w > 0.0f && (tile_idx * blockDim.x + j_tile) != i) {
                    float dx = p_j.x - pos_i_x; float dy = p_j.y - pos_i_y; float dz = p_j.z - pos_i_z;
                    float inv_dist = rsqrtf(dx*dx + dy*dy + dz*dz + eps*eps);
                    float forca = G * p_j.w * inv_dist * inv_dist * inv_dist; // p_j.w é a massa!
                    ax += forca * dx; ay += forca * dy; az += forca * dz;
                }
            }
            __syncthreads();
        }
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
"""

mod = SourceModule(kernel_code)
pre_update_gpu = mod.get_function("pre_update")
pre_update_float4_gpu = mod.get_function("pre_update_float4")
post_update_gpu = mod.get_function("post_update")

# Mapeamento de nomes de métodos para as funções de kernel compiladas.
# Esta estrutura permite adicionar novas otimizações facilmente.
# Basta adicionar um novo kernel C++ e mapear o seu nome aqui.
kernels_aceleracao = {
    "naive": mod.get_function("calcular_aceleracoes_naive"),
    "shared_mem": mod.get_function("calcular_aceleracoes_shared_mem"),
    "shared_mem_float4": mod.get_function("calcular_aceleracoes_shared_mem_float4"),
}

def simular_n_corpos_gpu(pos, vel, massas, passos, dt, G, eps, method: str = "naive"):
    N = pos.shape[0]

    import numpy as np # Mover import para dentro da função para evitar dependência a nível de módulo
    N_numpy = np.int32(N)
    
    vel_flat = vel.flatten().astype(np.float32)
    acel_flat = np.zeros(N * 3, dtype=np.float32)
    
    vel_gpu = cuda.mem_alloc(vel_flat.nbytes)
    acel_gpu = cuda.mem_alloc(acel_flat.nbytes)
    cuda.memcpy_htod(vel_gpu, vel_flat)
    cuda.memcpy_htod(acel_gpu, acel_flat)
    
    is_float4 = (method == "shared_mem_float4")

    if is_float4:
        # O Segredo do float4: Empacotar Posições (X,Y,Z) e Massas (W) no mesmo array NumPy de (N, 4)
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

    threads_por_bloco = 256
    blocos_por_grid = int(np.ceil(N / threads_por_bloco))
    block_dim = (threads_por_bloco, 1, 1)
    grid_dim = (blocos_por_grid, 1)

    if method not in kernels_aceleracao:
        raise ValueError(f"Método de kernel '{method}' desconhecido. Válidos: {list(kernels_aceleracao.keys())}")
    
    kernel_acel = kernels_aceleracao[method]

    # Aceleração inicial (t=0)
    if is_float4:
        kernel_acel(pos_gpu, acel_gpu, N_numpy, np.float32(G), np.float32(eps), block=block_dim, grid=grid_dim)
    else:
        kernel_acel(pos_gpu, massas_gpu, acel_gpu, N_numpy, np.float32(G), np.float32(eps), block=block_dim, grid=grid_dim)

    # Sincronizar e iniciar cronómetro só para o ciclo
    cuda.Context.synchronize()
    inicio_tempo = time.perf_counter()

    for _ in range(passos):
        if is_float4:
            pre_update_float4_gpu(pos_gpu, vel_gpu, acel_gpu, np.float32(dt), N_numpy, block=block_dim, grid=grid_dim)
            kernel_acel(pos_gpu, acel_gpu, N_numpy, np.float32(G), np.float32(eps), block=block_dim, grid=grid_dim)
        else:
            pre_update_gpu(pos_gpu, vel_gpu, acel_gpu, np.float32(dt), N_numpy, block=block_dim, grid=grid_dim)
            kernel_acel(pos_gpu, massas_gpu, acel_gpu, N_numpy, np.float32(G), np.float32(eps), block=block_dim, grid=grid_dim)
            
        post_update_gpu(vel_gpu, acel_gpu, np.float32(dt), N_numpy, block=block_dim, grid=grid_dim)

    cuda.Context.synchronize()
    fim_tempo = time.perf_counter()

    # Recuperar dados
    cuda.memcpy_dtoh(pos_flat, pos_gpu)
    
    if is_float4:
        # Extrair só os X,Y,Z das estruturas Float4 empacotadas
        pos_final = pos_flat.reshape((N, 4))[:, :3]
    else:
        pos_final = pos_flat.reshape((N, 3))

    return pos_final, (fim_tempo - inicio_tempo)