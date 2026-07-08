import numpy as np
import matplotlib.pyplot as plt
xi = np.array([0.038, 0.194, 0.425, 0.626, 1.253, 2.500, 3.740])
yi = np.array([0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317])


alpha = 0.9
beta = 0.2
max_iterations = 20
tolerance = 1e-10
functional_values_Gauss = []
functional_values_newton = []
for iteration in range(max_iterations):
    f_pred = alpha * xi / (beta + xi)
    ri = yi - f_pred
    S = np.sum(ri**2)
    #pour le graph
    functional_values_Gauss.append(S)
    functional_values_newton.append(S)
    
    
    J = np.zeros((len(xi), 2))
    denom = beta + xi               
    J[:, 0] = -xi / denom           
    J[:, 1] = alpha * xi / (denom**2)  
    
    grad = 2 * J.T @ ri
    JTJ = np.dot(J.T, J)           
    JTr = np.dot(J.T, ri)   
    try:
        delta = np.linalg.solve(JTJ, -JTr)  
    except np.linalg.LinAlgError:
        print("❌ Erreur : matrice singulière. Arrêt.")
        break
    
    delta_alpha, delta_beta = delta[0], delta[1]
    
    
    alpha_new = alpha + delta_alpha
    beta_new  = beta  + delta_beta

    

    alpha, beta = alpha_new, beta_new
plt.figure(figsize=(8, 5))
iterations = np.arange(len(functional_values_Gauss))
plt.semilogy(iterations, functional_values_Gauss, 'bo-', linewidth=2, markersize=6)
plt.xlabel('Itération', fontsize=12)
plt.ylabel(r' $S(\alpha, \beta) $', fontsize=12)
plt.title('Évolution de la fonctionnelle Gauss-Newton', fontsize=13)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()
print(f"v_max (alpha) = {alpha:.6f}")
print(f"K_M   (beta)  = {beta:.6f}")
print(f"Modèle ajusté : v(s) = ({alpha:.6f} * s) / ({beta:.6f} + s)")
# --- Méthode de Newton
alpha_n = 0.9
beta_n = 0.2
functional_values_newton = []

for iteration in range(max_iterations):
    f_pred = alpha_n * xi / (beta_n + xi)
    ri = yi - f_pred
    S = np.sum(ri**2)
    functional_values_newton.append(S)
    
    # Jacobienne
    denom = beta_n + xi
    J = np.zeros((len(xi), 2))
    J[:, 0] = -xi / denom
    J[:, 1] = alpha_n * xi / (denom**2)
    
    # Gradient
    grad = 2 * J.T @ ri  
    
    
    JTJ = J.T @ J  
    
    H_corr = np.zeros((2, 2))
    
    
    for i in range(len(xi)):
        d = beta_n + xi[i]
        r_i = ri[i]
        
        
        d2r_dadb = - xi[i] / (d**2)
        d2r_dbdb = - 2 * alpha_n * xi[i] / (d**3)
        
        H_r_i = np.array([[0.0, d2r_dadb],
                          [d2r_dadb, d2r_dbdb]])
        
        H_corr += r_i * H_r_i
    
    
    H = 2 * JTJ + 2 * H_corr
    
    
    try:
        delta = np.linalg.solve(H, -grad)
    except np.linalg.LinAlgError:
        print("❌ Newton : matrice hessienne singulière. Arrêt.")
        break
    
    alpha_n += delta[0]
    beta_n += delta[1]
    
   
    if np.linalg.norm(delta) < tolerance:
        functional_values_newton = functional_values_newton[:iteration+1]
        break

plt.figure(figsize=(9, 6))
iterations_gn = np.arange(len(functional_values_Gauss))
iterations_newton = np.arange(len(functional_values_newton))

plt.semilogy(iterations_gn, functional_values_Gauss, 'bo-', label='Gauss-Newton', linewidth=2, markersize=6)
plt.semilogy(iterations_newton, functional_values_newton, 'rs--', label='Newton (exact)', linewidth=2, markersize=6)

plt.xlabel('Itération', fontsize=12)
plt.ylabel(r'$S(\alpha, \beta)$', fontsize=12)
plt.title('Comparaison Gauss-Newton vs Newton (moindres carrés)', fontsize=13)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

print("\n=== Résultats finaux ===")
print(f"Gauss-Newton → v_max = {alpha:.6f}, K_M = {beta:.6f}")
print(f"Newton       → v_max = {alpha_n:.6f}, K_M = {beta_n:.6f}")

