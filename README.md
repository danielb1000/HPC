# Simulação N-Corpos (CPU vs GPU)

Este repositório contém uma implementação em Python de uma simulação física de N-Corpos (interação gravitacional) focada em High Performance Computing (HPC). O objetivo principal é demonstrar e comparar a performance entre a execução tradicional no processador (CPU) e a execução massivamente paralela acelerada por hardware na gráfica (GPU).

## Estrutura do Projeto e Relação entre Ficheiros

O projeto está desenhado de forma modular, separando a lógica de simulação em CPU, GPU e funções utilitárias.

### 🚀 Módulos Principais

* **`main.py`**
  O ponto de entrada principal (orquestrador). Ele gera as condições iniciais e, de seguida, corre a mesma simulação usando o CPU e a GPU. No fim, compara os tempos de execução (calculando o *speedup*), valida se os resultados matemáticos da GPU correspondem aos da CPU, e desenha o trajeto final das órbitas.
* **`simulacao_cpu.py`**
  Contém a implementação da simulação a correr no CPU. Utiliza o **NumPy** com operações vetorizadas e *broadcasting* para otimizar o cálculo de complexidade O(N²), evitando os lentos ciclos `for` em Python.
* **`simulacao_gpu.py`**
  Contém a implementação acelerada na gráfica. Utiliza o **PyCUDA** para compilar e executar código C++ (*Kernels*) na GPU, aproveitando milhares de *threads* para calcular as forças gravitacionais simultaneamente.

### 🛠️ Utilitários

* **`utilidades.py`**
  Fornece funções auxiliares partilhadas:
  - `gerar_condicoes_iniciais`: Inicializa matrizes de massas, posições e velocidades aleatórias.
  - `desenhar_grafico_n_corpos`: Pega no histórico de posições e usa o *Matplotlib* para projetar o rasto das partículas num gráfico 2D.

### 📚 Ficheiros de Estudo e Validação

* **`learning_cuda.py`**
  Um script monolítico de estudo que serviu para entender o funcionamento do PyCUDA. Contém toda a lógica e os *Kernels* em C++ num só ficheiro. Foi a base da qual o código de produção (`simulacao_gpu.py`) foi extraído.
* **`euler_vs_verlet.py`**
  Um script educativo que explica (visualmente) o problema de degradação orbital na integração numérica. Compara os métodos de *Euler* (diverge), *Euler-Cromer* (estável) e *Velocity Verlet* (muito preciso), justificando o uso do Verlet no projeto principal.
* **`simple2body.py`**
  Um script simples de validação (teste unitário) que calcula à mão as forças gravitacionais entre apenas 2 corpos em 3D, para garantir que as equações base da aceleração estão perfeitamente corretas antes de escalar para N corpos.

*(Se o número de partículas for baixo o suficiente, ou se a variável de visualização estiver ativa, uma janela abrir-se-á no final mostrando a trajetória dos corpos).*
