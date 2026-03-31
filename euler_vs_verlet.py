import math
import matplotlib.pyplot as plt

def calcular_aceleracao(pos_x, pos_y):
    # Massa central na origem (0,0). Constante GM = 1.0
    distancia_quadrada = pos_x**2 + pos_y**2
    distancia_cubo = math.sqrt(distancia_quadrada)**3
    
    # Cálculo das componentes do vetor aceleração: a = -pos / d^3
    ax, ay = -pos_x / distancia_cubo, -pos_y / distancia_cubo
    return ax, ay

# Condições iniciais (Órbita Circular perfeita)
# Todos começam exatamente no mesmo sítio (x=1.0, y=0.0) com a mesma velocidade (vx=0.0, vy=1.0)
pos_euler_x, pos_euler_y, vel_euler_x, vel_euler_y = 1.0, 0.0, 0.0, 1.0
pos_ec_x, pos_ec_y, vel_ec_x, vel_ec_y = 1.0, 0.0, 0.0, 1.0  # Para o Euler-Cromer
pos_verlet_x, pos_verlet_y, vel_verlet_x, vel_verlet_y = 1.0, 0.0,  0.0, 1.0

dt, passos = 0.05, 1000

# Listas para guardar o histórico de posições de cada método
hist_euler_x, hist_euler_y = [], []
hist_ec_x, hist_ec_y = [], []
hist_verlet_x, hist_verlet_y = [], []

acel_verlet_x, acel_verlet_y = calcular_aceleracao(pos_verlet_x, pos_verlet_y)

# Loop de Integração Temporal
for _ in range(passos):
    # Guardar posições atuais no histórico
    hist_euler_x.append(pos_euler_x)
    hist_euler_y.append(pos_euler_y)
    hist_ec_x.append(pos_ec_x)
    hist_ec_y.append(pos_ec_y)
    hist_verlet_x.append(pos_verlet_x)
    hist_verlet_y.append(pos_verlet_y)
    
    # --- 1. MÉTODO DE EULER PADRÃO ---
    acel_euler_x, acel_euler_y = calcular_aceleracao(pos_euler_x, pos_euler_y)
    
    # ERRO FATAL DO EULER: Atualiza posição com a velocidade ANTIGA
    pos_euler_x, pos_euler_y = pos_euler_x + vel_euler_x * dt, pos_euler_y + vel_euler_y * dt
    vel_euler_x, vel_euler_y = vel_euler_x + acel_euler_x * dt, vel_euler_y + acel_euler_y * dt
    

    # --- 2. MÉTODO EULER-CROMER ---
    acel_ec_x, acel_ec_y = calcular_aceleracao(pos_ec_x, pos_ec_y)
    
    # O "TRUQUE" DO EULER-CROMER: Atualiza a velocidade PRIMEIRO
    vel_ec_x, vel_ec_y = vel_ec_x + acel_ec_x * dt, vel_ec_y + acel_ec_y * dt
    # E de seguida usa a NOVA velocidade para atualizar a posição
    pos_ec_x, pos_ec_y = pos_ec_x + vel_ec_x * dt, pos_ec_y + vel_ec_y * dt
    

    # --- 3. MÉTODO VELOCITY VERLET ---
    # Passo 1: Atualizar posição com a velocidade e aceleração atuais
    pos_verlet_x += vel_verlet_x * dt + 0.5 * acel_verlet_x * (dt**2)
    pos_verlet_y += vel_verlet_y * dt + 0.5 * acel_verlet_y * (dt**2)
    
    # Passo 2: Calcular a nova aceleração na nova posição
    nova_acel_verlet_x, nova_acel_verlet_y = calcular_aceleracao(pos_verlet_x, pos_verlet_y)
    
    # Passo 3: Corrigir a velocidade combinando a aceleração antiga e a nova
    vel_verlet_x += 0.5 * (acel_verlet_x + nova_acel_verlet_x) * dt
    vel_verlet_y += 0.5 * (acel_verlet_y + nova_acel_verlet_y) * dt
    
    # Atualizar a aceleração para o ciclo seguinte
    acel_verlet_x, acel_verlet_y = nova_acel_verlet_x, nova_acel_verlet_y

# --- RENDERIZAÇÃO DO GRÁFICO ---
plt.figure(figsize=(8, 8))
plt.plot(hist_euler_x, hist_euler_y, 'r--', alpha=0.6, label="Euler Padrão (Diverge rapidamente)")
plt.plot(hist_ec_x, hist_ec_y, 'g-.', alpha=0.8, label="Euler-Cromer (Estável, conserva energia)")
plt.plot(hist_verlet_x, hist_verlet_y, 'b-', linewidth=2, label="Velocity Verlet (Mais preciso e estável)")
plt.plot(0, 0, 'yo', markersize=12, label="Massa Central (GM=1)")

plt.title("Estudo Numérico: Degradação Orbital por Método de Integração", fontsize=14)
plt.legend(loc="upper left")
plt.grid(True, linestyle=':', alpha=0.7)
plt.axis('equal')
plt.xlabel("Coordenada X")
plt.ylabel("Coordenada Y")
plt.tight_layout()
plt.xlim(-1.7, 1.7)
plt.ylim(-1.7, 1.7)
plt.savefig('comparacao_integradores.png', dpi=300)


plt.show()