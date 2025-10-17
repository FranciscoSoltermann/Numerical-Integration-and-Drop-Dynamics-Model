import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, RK45
import pandas as pd
import time

# ==================== DATOS EXPERIMENTALES ====================
# Cargar datos del TP4 (centro de masa en Y)
# Asumiendo que ya tienes los datos de centros_y del parte1.py
# Si no, carga desde archivo o calcula nuevamente

# Parámetros físicos
rho = 7380        # densidad [kg/m^3]
scale = 4.13e-6   # [m/px]
fps = 20538       # [frames/s]
dt_exp = 1.0 / fps  # paso de tiempo experimental

# Cargar o simular datos experimentales
# Aquí debes cargar los datos reales de centros_y del parte1.py
# Por ahora simularé datos de ejemplo
# IMPORTANTE: Reemplaza esto con tus datos reales

# Ejemplo de carga (ajusta según tu caso):
try:
    # Intenta cargar desde el análisis previo
    df_exp = pd.read_csv('resultados_volumen_area.csv')
    # Necesitamos también los centros Y del parte1
    # Si no están, hay que ejecutar parte1 primero y guardar
    print("Carga tus datos experimentales de centros_y aquí")
except:
    print("No se encontraron datos previos. Ejecuta primero parte1.py")

# Por ahora usaré datos de ejemplo para demostración
# REEMPLAZA ESTO con tus datos reales
n_frames = 100
tiempos_exp = np.arange(n_frames) * dt_exp
# Simulo una caída amortiguada como ejemplo
centros_y_exp = 50 + 30 * np.exp(-tiempos_exp * 5000) * np.cos(tiempos_exp * 10000)
centros_y_exp_m = centros_y_exp * scale  # convertir a metros

# Masa estimada (usar promedio de volúmenes calculados)
V_promedio = 5e-12  # [m^3] - AJUSTAR CON TUS DATOS
m = rho * V_promedio

# ==================== MODELO DINÁMICO ====================
def modelo_gota(t, state, m, k, c, y_eq):
    """
    Ecuación diferencial de segundo orden:
    m * d²y/dt² + c * dy/dt + k(y - y_eq) = 0
    
    Convertida a sistema de primer orden:
    dy/dt = v
    dv/dt = -(c/m)*v - (k/m)*(y - y_eq)
    """
    y, v = state
    dydt = v
    dvdt = -(c/m) * v - (k/m) * (y - y_eq)
    return [dydt, dvdt]

# ==================== a) MÉTODO TAYLOR ORDEN 3 + RUNGE-KUTTA 5-6 ====================
def taylor_orden3(f, t0, y0, h, args):
    """
    Método de Taylor de orden 3
    y(t+h) ≈ y(t) + h*f + (h²/2)*f' + (h³/6)*f''
    """
    # Evaluación en t0
    k1 = np.array(f(t0, y0, *args))
    
    # Aproximación de la derivada
    eps = 1e-8
    y_eps = y0 + eps * k1
    k2 = np.array(f(t0 + eps, y_eps, *args))
    f_prime = (k2 - k1) / eps
    
    # Segunda derivada
    y_eps2 = y0 + eps * k2
    k3 = np.array(f(t0 + 2*eps, y_eps2, *args))
    f_double_prime = (k3 - 2*k2 + k1) / (eps**2)
    
    # Taylor orden 3
    y_new = y0 + h * k1 + (h**2 / 2) * f_prime + (h**3 / 6) * f_double_prime
    return y_new

def resolver_taylor3(f, t_span, y0, args, n_steps):
    """Resuelve EDO con Taylor orden 3"""
    t0, tf = t_span
    h = (tf - t0) / n_steps
    
    t_vals = np.linspace(t0, tf, n_steps + 1)
    y_vals = np.zeros((n_steps + 1, len(y0)))
    y_vals[0] = y0
    
    for i in range(n_steps):
        y_vals[i + 1] = taylor_orden3(f, t_vals[i], y_vals[i], h, args)
    
    return t_vals, y_vals

def resolver_rk45(f, t_span, y0, args, rtol=1e-6, atol=1e-9):
    """Resuelve EDO con Runge-Kutta 4-5 adaptativo"""
    sol = solve_ivp(
        lambda t, y: f(t, y, *args),
        t_span,
        y0,
        method='RK45',
        rtol=rtol,
        atol=atol,
        dense_output=True
    )
    return sol

# ==================== ESTIMACIÓN DE PARÁMETROS ====================
def ajustar_parametros(tiempos_exp, y_exp, m, parametros_iniciales):
    """
    Ajusta k, c, y_eq para minimizar el error con datos experimentales
    """
    from scipy.optimize import minimize
    
    def error_modelo(params):
        k, c, y_eq = params
        if k <= 0 or c <= 0:
            return 1e10
        
        # Condiciones iniciales
        y0 = y_exp[0]
        v0 = (y_exp[1] - y_exp[0]) / (tiempos_exp[1] - tiempos_exp[0])
        state0 = [y0, v0]
        
        # Resolver
        sol = solve_ivp(
            lambda t, y: modelo_gota(t, y, m, k, c, y_eq),
            [tiempos_exp[0], tiempos_exp[-1]],
            state0,
            t_eval=tiempos_exp,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        
        if not sol.success:
            return 1e10
        
        y_modelo = sol.y[0]
        error = np.sum((y_modelo - y_exp)**2)
        return error
    
    resultado = minimize(
        error_modelo,
        parametros_iniciales,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8}
    )
    
    return resultado.x, resultado.fun

print("="*80)
print("AJUSTE DE PARÁMETROS DEL MODELO")
print("="*80)

# Estimaciones iniciales
k_inicial = 1.0  # rigidez efectiva [N/m]
c_inicial = 1e-6  # amortiguamiento [N·s/m]
y_eq_inicial = centros_y_exp_m[-10:].mean()  # altura de equilibrio

params_iniciales = [k_inicial, c_inicial, y_eq_inicial]
params_optimos, error_final = ajustar_parametros(
    tiempos_exp, centros_y_exp_m, m, params_iniciales
)

k_opt, c_opt, y_eq_opt = params_optimos

print(f"\nParámetros óptimos encontrados:")
print(f"  k (rigidez efectiva) = {k_opt:.6e} N/m")
print(f"  c (amortiguamiento)  = {c_opt:.6e} N·s/m")
print(f"  y_eq (altura equilibrio) = {y_eq_opt:.6e} m = {y_eq_opt/scale:.2f} px")
print(f"  Error cuadrático total = {error_final:.6e}")

# ==================== b) MÉTODO MULTIPASO ====================
def adams_bashforth_4(f, t_vals, y_vals, args):
    """
    Adams-Bashforth de 4 pasos (orden 4)
    Requiere 4 valores iniciales (usar RK4 para arrancar)
    """
    n = len(t_vals)
    h = t_vals[1] - t_vals[0]
    
    # Calcular valores de f en los primeros 4 puntos
    f_vals = [np.array(f(t_vals[i], y_vals[i], *args)) for i in range(4)]
    
    for i in range(3, n - 1):
        # Adams-Bashforth orden 4
        y_new = y_vals[i] + (h / 24) * (
            55 * f_vals[i] - 59 * f_vals[i-1] + 37 * f_vals[i-2] - 9 * f_vals[i-3]
        )
        y_vals[i + 1] = y_new
        f_vals.append(np.array(f(t_vals[i + 1], y_new, *args)))
    
    return y_vals

def resolver_adams_bashforth(f, t_span, y0, args, n_steps):
    """Resuelve EDO con Adams-Bashforth 4"""
    t0, tf = t_span
    h = (tf - t0) / n_steps
    
    t_vals = np.linspace(t0, tf, n_steps + 1)
    y_vals = np.zeros((n_steps + 1, len(y0)))
    y_vals[0] = y0
    
    # Usar RK4 para los primeros 3 pasos
    from scipy.integrate import RK45
    solver = RK45(lambda t, y: f(t, y, *args), t0, y0, tf, max_step=h)
    
    for i in range(1, 4):
        solver.step()
        y_vals[i] = solver.y
        t_vals[i] = solver.t
    
    # Continuar con Adams-Bashforth
    y_vals = adams_bashforth_4(f, t_vals, y_vals, args)
    
    return t_vals, y_vals

# ==================== c) COMPARACIÓN DE TIEMPOS COMPUTACIONALES ====================
print("\n" + "="*80)
print("COMPARACIÓN DE MÉTODOS NUMÉRICOS")
print("="*80)

# Condiciones iniciales
y0_m = centros_y_exp_m[0]
v0_m = (centros_y_exp_m[1] - centros_y_exp_m[0]) / dt_exp
state0 = [y0_m, v0_m]

t_span = [tiempos_exp[0], tiempos_exp[-1]]
args = (m, k_opt, c_opt, y_eq_opt)

# Diferentes tolerancias para probar
tolerancias = [1e-3, 1e-6, 1e-9]
n_steps_taylor = [100, 500, 1000, 5000]

resultados_timing = []

# 1) Taylor orden 3
print("\n--- Método de Taylor orden 3 ---")
for n_steps in n_steps_taylor:
    t_inicio = time.time()
    t_taylor, y_taylor = resolver_taylor3(modelo_gota, t_span, state0, args, n_steps)
    t_fin = time.time()
    tiempo_ejec = t_fin - t_inicio
    
    # Interpolar para comparar
    y_taylor_interp = np.interp(tiempos_exp, t_taylor, y_taylor[:, 0])
    error = np.sqrt(np.mean((y_taylor_interp - centros_y_exp_m)**2))
    
    print(f"  n_steps={n_steps:5d}: Tiempo={tiempo_ejec:.6f}s, RMSE={error:.6e}m")
    resultados_timing.append({
        'Método': 'Taylor-3',
        'Parámetro': f'n={n_steps}',
        'Tiempo_s': tiempo_ejec,
        'RMSE_m': error
    })

# 2) Runge-Kutta 4-5
print("\n--- Método Runge-Kutta 4-5 (adaptativo) ---")
for tol in tolerancias:
    t_inicio = time.time()
    sol_rk45 = resolver_rk45(modelo_gota, t_span, state0, args, rtol=tol, atol=tol*1e-3)
    t_fin = time.time()
    tiempo_ejec = t_fin - t_inicio
    
    y_rk45 = sol_rk45.sol(tiempos_exp)[0]
    error = np.sqrt(np.mean((y_rk45 - centros_y_exp_m)**2))
    
    print(f"  tol={tol:.0e}: Tiempo={tiempo_ejec:.6f}s, RMSE={error:.6e}m, Evaluaciones={sol_rk45.nfev}")
    resultados_timing.append({
        'Método': 'RK45',
        'Parámetro': f'tol={tol:.0e}',
        'Tiempo_s': tiempo_ejec,
        'RMSE_m': error
    })

# 3) Adams-Bashforth
print("\n--- Método Adams-Bashforth 4 (multipaso) ---")
for n_steps in [500, 1000, 5000]:
    t_inicio = time.time()
    t_ab, y_ab = resolver_adams_bashforth(modelo_gota, t_span, state0, args, n_steps)
    t_fin = time.time()
    tiempo_ejec = t_fin - t_inicio
    
    y_ab_interp = np.interp(tiempos_exp, t_ab, y_ab[:, 0])
    error = np.sqrt(np.mean((y_ab_interp - centros_y_exp_m)**2))
    
    print(f"  n_steps={n_steps:5d}: Tiempo={tiempo_ejec:.6f}s, RMSE={error:.6e}m")
    resultados_timing.append({
        'Método': 'Adams-Bashforth-4',
        'Parámetro': f'n={n_steps}',
        'Tiempo_s': tiempo_ejec,
        'RMSE_m': error
    })

df_timing = pd.DataFrame(resultados_timing)

# ==================== d) COMPARACIÓN CON DATOS EXPERIMENTALES ====================
# Resolver con los mejores parámetros usando RK45
sol_final = resolver_rk45(modelo_gota, t_span, state0, args, rtol=1e-9, atol=1e-12)
y_modelo_final = sol_final.sol(tiempos_exp)[0]
v_modelo_final = sol_final.sol(tiempos_exp)[1]

# Calcular desviaciones
desviaciones = y_modelo_final - centros_y_exp_m
desv_relativa = desviaciones / centros_y_exp_m * 100

# ==================== VISUALIZACIONES ====================
plt.style.use('dark_background')

# Gráfico 1: Comparación modelo vs experimental
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.plot(tiempos_exp * 1e6, centros_y_exp_m * 1e6, 'o', 
         label='Datos experimentales', markersize=4, alpha=0.7)
ax1.plot(tiempos_exp * 1e6, y_modelo_final * 1e6, '-', 
         label='Modelo ajustado', linewidth=2)
ax1.set_xlabel('Tiempo [µs]', fontsize=12)
ax1.set_ylabel('Altura centro de masa [µm]', fontsize=12)
ax1.set_title('Comparación: Modelo vs Datos Experimentales', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(tiempos_exp * 1e6, desv_relativa, 'r-', linewidth=2)
ax2.axhline(0, color='white', linestyle='--', alpha=0.5)
ax2.set_xlabel('Tiempo [µs]', fontsize=12)
ax2.set_ylabel('Desviación relativa [%]', fontsize=12)
ax2.set_title('Error Relativo del Modelo', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Gráfico 2: Velocidad del centro de masa
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(tiempos_exp * 1e6, v_modelo_final, 'g-', linewidth=2, label='Velocidad modelada')
ax.set_xlabel('Tiempo [µs]', fontsize=12)
ax.set_ylabel('Velocidad [m/s]', fontsize=12)
ax.set_title('Velocidad del Centro de Masa', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Gráfico 3: Comparación de tiempos computacionales
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for metodo in df_timing['Método'].unique():
    df_metodo = df_timing[df_timing['Método'] == metodo]
    ax1.plot(range(len(df_metodo)), df_metodo['Tiempo_s'], 'o-', 
             label=metodo, markersize=8, linewidth=2)

ax1.set_xlabel('Configuración', fontsize=12)
ax1.set_ylabel('Tiempo de ejecución [s]', fontsize=12)
ax1.set_title('Comparación de Tiempos Computacionales', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

for metodo in df_timing['Método'].unique():
    df_metodo = df_timing[df_timing['Método'] == metodo]
    ax2.plot(range(len(df_metodo)), df_metodo['RMSE_m'] * 1e6, 's-', 
             label=metodo, markersize=8, linewidth=2)

ax2.set_xlabel('Configuración', fontsize=12)
ax2.set_ylabel('RMSE [µm]', fontsize=12)
ax2.set_title('Precisión de los Métodos', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== ANÁLISIS FÍSICO ====================
print("\n" + "="*80)
print("ANÁLISIS DE RESULTADOS")
print("="*80)

print(f"\nEstadísticas del ajuste:")
print(f"  RMSE = {np.sqrt(np.mean(desviaciones**2)) * 1e6:.4f} µm")
print(f"  Error relativo promedio = {np.mean(np.abs(desv_relativa)):.4f}%")
print(f"  Error relativo máximo = {np.max(np.abs(desv_relativa)):.4f}%")

print(f"\nAnálisis del sistema dinámico:")
omega_n = np.sqrt(k_opt / m)
zeta = c_opt / (2 * np.sqrt(k_opt * m))
print(f"  Frecuencia natural: ω_n = {omega_n:.2f} rad/s = {omega_n/(2*np.pi):.2f} Hz")
print(f"  Factor de amortiguamiento: ζ = {zeta:.6f}")
if zeta < 1:
    print(f"  Sistema SUBAMORTIGUADO (oscilatorio)")
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    print(f"  Frecuencia amortiguada: ω_d = {omega_d:.2f} rad/s")
elif zeta == 1:
    print(f"  Sistema CRÍTICAMENTE AMORTIGUADO")
else:
    print(f"  Sistema SOBREAMORTIGUADO")

print(f"\nPosibles causas de desviaciones:")
print(f"  1. Simplificación del modelo (centro de masa como partícula)")
print(f"  2. Parámetros k y c pueden variar durante el spreading")
print(f"  3. Efectos de tensión superficial no incluidos explícitamente")
print(f"  4. Interacción compleja con el sustrato")
print(f"  5. Efectos viscosos internos del líquido")

# Guardar resultados
df_timing.to_csv('comparacion_metodos_numericos.csv', index=False)
print("\n✓ Resultados guardados en 'comparacion_metodos_numericos.csv'")

# Guardar comparación modelo-experimental
df_comparacion = pd.DataFrame({
    'Tiempo_us': tiempos_exp * 1e6,
    'Y_experimental_um': centros_y_exp_m * 1e6,
    'Y_modelo_um': y_modelo_final * 1e6,
    'Desviacion_um': desviaciones * 1e6,
    'Desviacion_relativa_%': desv_relativa,
    'Velocidad_m_s': v_modelo_final
})
df_comparacion.to_csv('modelo_vs_experimental.csv', index=False)
print("✓ Comparación guardada en 'modelo_vs_experimental.csv'")