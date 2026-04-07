# Simulação N-Corpos (CPU vs GPU)

Este repositório contém uma implementação em Python de uma simulação física de N-Corpos (interação gravitacional) focada em High Performance Computing (HPC). O objetivo principal é demonstrar e comparar a performance entre a execução tradicional no processador (CPU) e a execução massivamente paralela acelerada por hardware na gráfica (GPU), evoluindo até arquiteturas Multi-GPU com suporte NVLink.

## Estrutura do Projeto e Relação entre Ficheiros

O projeto está desenhado de forma modular, separando a lógica de simulação em CPU, várias etapas de otimização em GPU (Single e Multi-GPU) e funções utilitárias.

### 🚀 Módulos Principais

* **`main.py`**
  O ponto de entrada principal (orquestrador e de *benchmarking*). Gera as condições iniciais e corre sucessivas simulações progressivamente mais otimizadas (CPU -> GPU Naive -> Shared Memory -> Float4 -> Multi-GPU -> NVLink), validando resultados, calculando *speedups* e, opcionalmente, desenhando as órbitas.
* **`simulacao_cpu.py`**
  Implementação no CPU utilizando **NumPy** com operações vetorizadas e *broadcasting* para otimizar o cálculo de complexidade O(N²), evitando os lentos ciclos `for` em Python.
* **`simulacao_gpu.py`**
  Implementações baseadas em **PyCUDA** para uma única GPU. Demonstra a evolução das otimizações na VRAM: baseline (*Naive*), uso de funções intrínsecas (*Fast Math*), aproveitamento de memória partilhada (*Shared Memory/Tiling*) e vetorização de leituras de memória (*Float4*).
* **`simulacao_multigpu.py`**
  Implementação distribuída para múltiplas placas gráficas usando `multiprocessing`. Divide o universo de partículas pelas GPUs disponíveis, sincronizando os dados através de *Shared Memory* no CPU (Host) a cada passo de tempo.
* **`simulacao_nv4.py`**
  Implementação Multi-GPU avançada que tira partido da tecnologia **NVLink/NVSwitch**. Em vez de usar a RAM do CPU como ponte intermediária, as GPUs partilham ponteiros e acedem diretamente à VRAM umas das outras (P2P - *Device-to-Device*), eliminando drásticamente o gargalo do barramento PCIe.

### 🛠️ Utilitários e Diagnóstico

* **`utilidades.py`**
  Fornece funções auxiliares partilhadas:
  - `gerar_condicoes_iniciais`: Inicializa matrizes de massas, posições e velocidades aleatórias.
  - `desenhar_grafico_n_corpos`: Pega no histórico de posições e usa o *Matplotlib* para projetar o rasto das partículas num gráfico 2D.
* **`teste_servidor_gpu.py`**
  Script de diagnóstico concebido para testar a inicialização concorrente, gestão de contextos PyCUDA e a alocação de memória em servidores HPC com múltiplas GPUs.

### 📚 Ficheiros de Estudo e Validação

* **`learning_cuda.py`**
  Um script monolítico de estudo que serviu para entender o funcionamento do PyCUDA. Contém toda a lógica e os *Kernels* em C++ num só ficheiro. Foi a base da qual o código de produção (`simulacao_gpu.py`) foi extraído.
* **`euler_vs_verlet.py`**
  Um script educativo que explica (visualmente) o problema de degradação orbital na integração numérica. Compara os métodos de *Euler* (diverge), *Euler-Cromer* (estável) e *Velocity Verlet* (muito preciso), justificando o uso do Verlet no projeto principal.
* **`simple2body.py`**
  Um script simples de validação (teste unitário) que calcula à mão as forças gravitacionais entre apenas 2 corpos em 3D, para garantir que as equações base da aceleração estão perfeitamente corretas antes de escalar para N corpos.

*(Se o número de partículas for baixo o suficiente, ou se a variável de visualização estiver ativa, uma janela abrir-se-á no final mostrando a trajetória dos corpos).*
