def calcular_aceleracao(m_j, pos_i, pos_j, G=1.0, eps=0.0):
    # 1. Distância vetorial (r_j - r_i)
    dx, dy, dz = pos_j[0] - pos_i[0], pos_j[1] - pos_i[1], pos_j[2] - pos_i[2]
    # 2. Quadrado da distância
    dist_sq = dx**2 + dy**2 + dz**2
    # 3. Denominador da fórmula: (||r||^2 + eps^2)^(3/2)
    denominador = (dist_sq + eps**2) ** 1.5
    # 4. Cálculo final da aceleração nos 3 eixos
    return G*m_j*dx/denominador, G*m_j*dy/denominador, G*m_j*dz/denominador

    
# --- TESTES DE VALIDAÇÃO ---
pos_p1 = (0, 0, 0)

# Teste 1: Caso Base (Distância=10, Massa=1000)
pos_p2_base = (10, 0, 0)
a_base = calcular_aceleracao(1000, pos_p1, pos_p2_base)[0] # Olhando só para o eixo X
print(f"1. Aceleração Base: {a_base}") # Resultado: 10.0

# Teste 2: Dobro da Massa (Distância=10, Massa=2000)
a_massa = calcular_aceleracao(2000, pos_p1, pos_p2_base)[0]
print(f"2. Massa Duplicada: {a_massa}") # Resultado: 20.0 (O dobro da base)

# Teste 3: Metade da Distância (Distância=5, Massa=1000)
pos_p2_perto = (5, 0, 0)
a_dist = calcular_aceleracao(1000, pos_p1, pos_p2_perto)[0]
print(f"3. Distância a Metade: {a_dist}") # Resultado: 40.0 (O quádruplo da base)