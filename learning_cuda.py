import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
#
#
# Ficheiro básico de aprendizagem CUDA gerado por IA.
# 
#


# O nosso código C++ puro que vai correr na GPU
kernel_code = """
__global__ void calcular_aceleracoes(float *pos, float *massas, float *acel, int N, float G, float eps) {
    
    // 1. Quem sou eu? (Fórmula mágica do CUDA para descobrir o ID da thread)
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // 2. Proteção: Se tivermos mais threads que partículas, as threads extra não fazem nada
    if (i < N) {
        float ax = 0.0f;
        float ay = 0.0f;
        float az = 0.0f;

        // Extrair as posições X, Y e Z da minha partícula 'i'
        // Como o array em C é unidimensional, multiplicamos por 3
        float pos_i_x = pos[i * 3 + 0];
        float pos_i_y = pos[i * 3 + 1];
        float pos_i_z = pos[i * 3 + 2];

        // 3. Iterar sobre todas as outras N partículas (O ciclo clássico)
        for (int j = 0; j < N; j++) {
            if (i != j) { // Não calculo a atração de mim próprio!
                
                float dx = pos[j * 3 + 0] - pos_i_x;
                float dy = pos[j * 3 + 1] - pos_i_y;
                float dz = pos[j * 3 + 2] - pos_i_z;

                float dist_sq = dx*dx + dy*dy + dz*dz;
                
                // powf é a função C++ nativa para elevar à potência usando floats
                float denominador = powf(dist_sq + eps*eps, 1.5f);

                float forca = (G * massas[j]) / denominador;

                ax += forca * dx;
                ay += forca * dy;
                az += forca * dz;
            }
        }

        // 4. Escrever o resultado final na memória global da GPU
        acel[i * 3 + 0] = ax;
        acel[i * 3 + 1] = ay;
        acel[i * 3 + 2] = az;
    }
}
"""

# Compilar o kernel
mod = SourceModule(kernel_code)
calcular_aceleracoes_gpu = mod.get_function("calcular_aceleracoes")

