import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson, trapezoid
import pandas as pd

# ==================== CONFIGURACIÓN ====================
folder = "TP4_imagenes"
frame_files = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])

# Parámetros
y_sustrato = 130
y_contact_line = 127
spline_smooth = 0.5
degree_candidates = [3, 4, 5, 6, 7]

# Constantes físicas
scale = 4.13e-6   # [m/px]

# ==================== FUNCIONES AUXILIARES ====================
def best_poly_degree(x, y, degrees):
    """Encuentra el mejor grado polinomial por mínimos cuadrados"""
    best_deg = degrees[0]
    min_mse = float('inf')
    for deg in degrees:
        p = np.poly1d(np.polyfit(y, x, deg))
        mse = np.mean((x - p(y))**2)
        if mse < min_mse:
            min_mse = mse
            best_deg = deg
    return best_deg

def remove_duplicate_y(y, x):
    """Elimina duplicados en Y promediando los valores de X"""
    y_unique, x_avg = [], []
    for val in np.unique(y):
        mask = (y == val)
        y_unique.append(val)
        x_avg.append(np.mean(x[mask]))
    return np.array(y_unique), np.array(x_avg)

# ==================== MÉTODOS DE INTEGRACIÓN PARA VOLUMEN ====================
def volumen_simpson_spline(y_left, x_left, y_right, x_right):
    """Método 1: Simpson con splines"""
    y_min = max(y_left.min(), y_right.min())
    y_max = min(y_left.max(), y_right.max())
    y_common = np.linspace(y_min, y_max, 500)
    
    spline_left = UnivariateSpline(y_left, x_left, s=spline_smooth)
    spline_right = UnivariateSpline(y_right, x_right, s=spline_smooth)
    
    xL = spline_left(y_common)
    xR = spline_right(y_common)
    radios = (xR - xL) / 2.0
    
    integrando = np.pi * radios**2
    vol_px3 = simpson(integrando, x=y_common)
    return vol_px3 * (scale**3)

def volumen_trapecio_spline(y_left, x_left, y_right, x_right):
    """Método 2: Trapezoidal con splines"""
    y_min = max(y_left.min(), y_right.min())
    y_max = min(y_left.max(), y_right.max())
    y_common = np.linspace(y_min, y_max, 500)
    
    spline_left = UnivariateSpline(y_left, x_left, s=spline_smooth)
    spline_right = UnivariateSpline(y_right, x_right, s=spline_smooth)
    
    xL = spline_left(y_common)
    xR = spline_right(y_common)
    radios = (xR - xL) / 2.0
    
    integrando = np.pi * radios**2
    vol_px3 = trapezoid(integrando, y_common)
    return vol_px3 * (scale**3)

def volumen_simpson_poly(y_left, x_left, y_right, x_right, degree_left, degree_right):
    """Método 3: Simpson con polinomios"""
    y_min = max(y_left.min(), y_right.min())
    y_max = min(y_left.max(), y_right.max())
    y_common = np.linspace(y_min, y_max, 500)
    
    poly_left = np.poly1d(np.polyfit(y_left, x_left, degree_left))
    poly_right = np.poly1d(np.polyfit(y_right, x_right, degree_right))
    
    xL = poly_left(y_common)
    xR = poly_right(y_common)
    radios = (xR - xL) / 2.0
    
    integrando = np.pi * radios**2
    vol_px3 = simpson(integrando, x=y_common)
    return vol_px3 * (scale**3)

def volumen_trapecio_poly(y_left, x_left, y_right, x_right, degree_left, degree_right):
    """Método 4: Trapezoidal con polinomios"""
    y_min = max(y_left.min(), y_right.min())
    y_max = min(y_left.max(), y_right.max())
    y_common = np.linspace(y_min, y_max, 500)
    
    poly_left = np.poly1d(np.polyfit(y_left, x_left, degree_left))
    poly_right = np.poly1d(np.polyfit(y_right, x_right, degree_right))
    
    xL = poly_left(y_common)
    xR = poly_right(y_common)
    radios = (xR - xL) / 2.0
    
    integrando = np.pi * radios**2
    vol_px3 = trapezoid(integrando, y_common)
    return vol_px3 * (scale**3)

# ==================== MÉTODOS DE INTEGRACIÓN PARA ÁREA SUPERFICIAL ====================
def area_simpson_spline(y_left, x_left, y_right, x_right):
    """Método 1: Simpson con splines para área superficial"""
    y_min = max(y_left.min(), y_right.min())
    y_max = min(y_left.max(), y_right.max())
    y_common = np.linspace(y_min, y_max, 500)
    
    spline_left = UnivariateSpline(y_left, x_left, s=spline_smooth)
    spline_right = UnivariateSpline(y_right, x_right, s=spline_smooth)
    
    xL = spline_left(y_common)
    xR = spline_right(y_common)
    dxL_dy = spline_left.derivative()(y_common)
    dxR_dy = spline_right.derivative()(y_common)
    
    radios = (xR - xL) / 2.0
    
    # Lado izquierdo
    ds_left = np.sqrt(1 + dxL_dy**2)
    integrando_left = 2 * np.pi * radios * ds_left
    area_left = simpson(integrando_left, x=y_common)
    
    # Lado derecho
    ds_right = np.sqrt(1 + dxR_dy**2)
    integrando_right = 2 * np.pi * radios * ds_right
    area_right = simpson(integrando_right, x=y_common)
    
    area_total_px2 = (area_left + area_right) / 2
    return area_total_px2 * (scale**2)

def area_trapecio_spline(y_left, x_left, y_right, x_right):
    """Método 2: Trapezoidal con splines para área superficial"""
    y_min = max(y_left.min(), y_right.min())
    y_max = min(y_left.max(), y_right.max())
    y_common = np.linspace(y_min, y_max, 500)
    
    spline_left = UnivariateSpline(y_left, x_left, s=spline_smooth)
    spline_right = UnivariateSpline(y_right, x_right, s=spline_smooth)
    
    xL = spline_left(y_common)
    xR = spline_right(y_common)
    dxL_dy = spline_left.derivative()(y_common)
    dxR_dy = spline_right.derivative()(y_common)
    
    radios = (xR - xL) / 2.0
    
    ds_left = np.sqrt(1 + dxL_dy**2)
    integrando_left = 2 * np.pi * radios * ds_left
    area_left = trapezoid(integrando_left, y_common)
    
    ds_right = np.sqrt(1 + dxR_dy**2)
    integrando_right = 2 * np.pi * radios * ds_right
    area_right = trapezoid(integrando_right, y_common)
    
    area_total_px2 = (area_left + area_right) / 2
    return area_total_px2 * (scale**2)

def area_simpson_poly(y_left, x_left, y_right, x_right, degree_left, degree_right):
    """Método 3: Simpson con polinomios para área superficial"""
    y_min = max(y_left.min(), y_right.min())
    y_max = min(y_left.max(), y_right.max())
    y_common = np.linspace(y_min, y_max, 500)
    
    poly_left = np.poly1d(np.polyfit(y_left, x_left, degree_left))
    poly_right = np.poly1d(np.polyfit(y_right, x_right, degree_right))
    
    xL = poly_left(y_common)
    xR = poly_right(y_common)
    dxL_dy = poly_left.deriv()(y_common)
    dxR_dy = poly_right.deriv()(y_common)
    
    radios = (xR - xL) / 2.0
    
    ds_left = np.sqrt(1 + dxL_dy**2)
    integrando_left = 2 * np.pi * radios * ds_left
    area_left = simpson(integrando_left, x=y_common)
    
    ds_right = np.sqrt(1 + dxR_dy**2)
    integrando_right = 2 * np.pi * radios * ds_right
    area_right = simpson(integrando_right, x=y_common)
    
    area_total_px2 = (area_left + area_right) / 2
    return area_total_px2 * (scale**2)

def area_trapecio_poly(y_left, x_left, y_right, x_right, degree_left, degree_right):
    """Método 4: Trapezoidal con polinomios para área superficial"""
    y_min = max(y_left.min(), y_right.min())
    y_max = min(y_left.max(), y_right.max())
    y_common = np.linspace(y_min, y_max, 500)
    
    poly_left = np.poly1d(np.polyfit(y_left, x_left, degree_left))
    poly_right = np.poly1d(np.polyfit(y_right, x_right, degree_right))
    
    xL = poly_left(y_common)
    xR = poly_right(y_common)
    dxL_dy = poly_left.deriv()(y_common)
    dxR_dy = poly_right.deriv()(y_common)
    
    radios = (xR - xL) / 2.0
    
    ds_left = np.sqrt(1 + dxL_dy**2)
    integrando_left = 2 * np.pi * radios * ds_left
    area_left = trapezoid(integrando_left, y_common)
    
    ds_right = np.sqrt(1 + dxR_dy**2)
    integrando_right = 2 * np.pi * radios * ds_right
    area_right = trapezoid(integrando_right, y_common)
    
    area_total_px2 = (area_left + area_right) / 2
    return area_total_px2 * (scale**2)

# ==================== PROCESAMIENTO DE FRAMES ====================
resultados = []

for idx, fname in enumerate(frame_files):
    img_path = os.path.join(folder, fname)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ROI hasta el sustrato
    roi = gray[:y_sustrato+1, :]
    _, binary = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        continue
    
    # Contorno más grande
    contour = max(contours, key=cv2.contourArea)[:, 0, :]
    x, y = contour[:, 0], contour[:, 1]
    mid_x = x.mean()
    
    # Separar lados
    left, right = contour[x < mid_x], contour[x >= mid_x]
    sort_left, sort_right = np.argsort(left[:, 1]), np.argsort(right[:, 1])
    y_left, x_left = left[:, 1][sort_left], left[:, 0][sort_left]
    y_right, x_right = right[:, 1][sort_right], right[:, 0][sort_right]
    
    # Eliminar duplicados
    y_left, x_left = remove_duplicate_y(y_left, x_left)
    y_right, x_right = remove_duplicate_y(y_right, x_right)
    
    # Determinar grados óptimos para polinomios
    deg_left = best_poly_degree(x_left, y_left, degree_candidates)
    deg_right = best_poly_degree(x_right, y_right, degree_candidates)
    
    # VOLÚMENES
    V_simp_spl = volumen_simpson_spline(y_left, x_left, y_right, x_right)
    V_trap_spl = volumen_trapecio_spline(y_left, x_left, y_right, x_right)
    V_simp_poly = volumen_simpson_poly(y_left, x_left, y_right, x_right, deg_left, deg_right)
    V_trap_poly = volumen_trapecio_poly(y_left, x_left, y_right, x_right, deg_left, deg_right)
    
    # ÁREAS SUPERFICIALES
    A_simp_spl = area_simpson_spline(y_left, x_left, y_right, x_right)
    A_trap_spl = area_trapecio_spline(y_left, x_left, y_right, x_right)
    A_simp_poly = area_simpson_poly(y_left, x_left, y_right, x_right, deg_left, deg_right)
    A_trap_poly = area_trapecio_poly(y_left, x_left, y_right, x_right, deg_left, deg_right)
    
    resultados.append({
        'Frame': idx + 1,
        'V_Simpson_Spline': V_simp_spl,
        'V_Trapecio_Spline': V_trap_spl,
        'V_Simpson_Poly': V_simp_poly,
        'V_Trapecio_Poly': V_trap_poly,
        'A_Simpson_Spline': A_simp_spl,
        'A_Trapecio_Spline': A_trap_spl,
        'A_Simpson_Poly': A_simp_poly,
        'A_Trapecio_Poly': A_trap_poly,
        'deg_left': deg_left,
        'deg_right': deg_right
    })
    
    print(f"Frame {idx+1}: V_simp_spl={V_simp_spl:.3e} m³, A_simp_spl={A_simp_spl:.3e} m²")

df = pd.DataFrame(resultados)

# ==================== ANÁLISIS DE ERRORES ====================
# Calcular promedios y errores relativos para cada frame
df['V_promedio'] = df[['V_Simpson_Spline', 'V_Trapecio_Spline', 'V_Simpson_Poly', 'V_Trapecio_Poly']].mean(axis=1)
df['A_promedio'] = df[['A_Simpson_Spline', 'A_Trapecio_Spline', 'A_Simpson_Poly', 'A_Trapecio_Poly']].mean(axis=1)

# Errores relativos respecto al promedio
for col in ['V_Simpson_Spline', 'V_Trapecio_Spline', 'V_Simpson_Poly', 'V_Trapecio_Poly']:
    df[f'Error_{col}'] = np.abs(df[col] - df['V_promedio']) / df['V_promedio'] * 100

for col in ['A_Simpson_Spline', 'A_Trapecio_Spline', 'A_Simpson_Poly', 'A_Trapecio_Poly']:
    df[f'Error_{col}'] = np.abs(df[col] - df['A_promedio']) / df['A_promedio'] * 100

# ==================== VISUALIZACIÓN ====================
plt.style.use('dark_background')

# Gráfico 1: Evolución de volúmenes
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Frame'], df['V_Simpson_Spline'], 'o-', label='Simpson + Spline', linewidth=2)
ax.plot(df['Frame'], df['V_Trapecio_Spline'], 's-', label='Trapecio + Spline', linewidth=2)
ax.plot(df['Frame'], df['V_Simpson_Poly'], '^-', label='Simpson + Polinomio', linewidth=2)
ax.plot(df['Frame'], df['V_Trapecio_Poly'], 'd-', label='Trapecio + Polinomio', linewidth=2)
ax.set_xlabel('Frame', fontsize=12)
ax.set_ylabel('Volumen [m³]', fontsize=12)
ax.set_title('Evolución del Volumen - Comparación de Métodos', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Gráfico 2: Evolución de áreas superficiales
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Frame'], df['A_Simpson_Spline'], 'o-', label='Simpson + Spline', linewidth=2)
ax.plot(df['Frame'], df['A_Trapecio_Spline'], 's-', label='Trapecio + Spline', linewidth=2)
ax.plot(df['Frame'], df['A_Simpson_Poly'], '^-', label='Simpson + Polinomio', linewidth=2)
ax.plot(df['Frame'], df['A_Trapecio_Poly'], 'd-', label='Trapecio + Polinomio', linewidth=2)
ax.set_xlabel('Frame', fontsize=12)
ax.set_ylabel('Área Superficial [m²]', fontsize=12)
ax.set_title('Evolución del Área Superficial - Comparación de Métodos', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Gráfico 3: Errores relativos de volumen
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Frame'], df['Error_V_Simpson_Spline'], 'o-', label='Simpson + Spline', linewidth=2)
ax.plot(df['Frame'], df['Error_V_Trapecio_Spline'], 's-', label='Trapecio + Spline', linewidth=2)
ax.plot(df['Frame'], df['Error_V_Simpson_Poly'], '^-', label='Simpson + Polinomio', linewidth=2)
ax.plot(df['Frame'], df['Error_V_Trapecio_Poly'], 'd-', label='Trapecio + Polinomio', linewidth=2)
ax.set_xlabel('Frame', fontsize=12)
ax.set_ylabel('Error Relativo [%]', fontsize=12)
ax.set_title('Error Relativo del Volumen respecto al Promedio', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Gráfico 4: Errores relativos de área
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Frame'], df['Error_A_Simpson_Spline'], 'o-', label='Simpson + Spline', linewidth=2)
ax.plot(df['Frame'], df['Error_A_Trapecio_Spline'], 's-', label='Trapecio + Spline', linewidth=2)
ax.plot(df['Frame'], df['Error_A_Simpson_Poly'], '^-', label='Simpson + Polinomio', linewidth=2)
ax.plot(df['Frame'], df['Error_A_Trapecio_Poly'], 'd-', label='Trapecio + Polinomio', linewidth=2)
ax.set_xlabel('Frame', fontsize=12)
ax.set_ylabel('Error Relativo [%]', fontsize=12)
ax.set_title('Error Relativo del Área Superficial respecto al Promedio', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==================== ESTADÍSTICAS FINALES ====================
print("\n" + "="*80)
print("RESUMEN ESTADÍSTICO DE ERRORES")
print("="*80)

print("\n--- VOLUMEN ---")
for col in ['Error_V_Simpson_Spline', 'Error_V_Trapecio_Spline', 'Error_V_Simpson_Poly', 'Error_V_Trapecio_Poly']:
    nombre = col.replace('Error_V_', '')
    print(f"{nombre:25s}: Error medio = {df[col].mean():.4f}%, Error máx = {df[col].max():.4f}%")

print("\n--- ÁREA SUPERFICIAL ---")
for col in ['Error_A_Simpson_Spline', 'Error_A_Trapecio_Spline', 'Error_A_Simpson_Poly', 'Error_A_Trapecio_Poly']:
    nombre = col.replace('Error_A_', '')
    print(f"{nombre:25s}: Error medio = {df[col].mean():.4f}%, Error máx = {df[col].max():.4f}%")

print("\n" + "="*80)
print("ANÁLISIS DE FUENTES DE ERROR")
print("="*80)
print("""
1. TIPO DE AJUSTE (Spline vs Polinomio):
   - Splines: Mayor suavidad local, mejor para contornos irregulares
   - Polinomios: Ajuste global, pueden oscilar en los extremos (fenómeno de Runge)

2. MÉTODO DE INTEGRACIÓN (Simpson vs Trapecio):
   - Simpson: Mayor precisión (error O(h⁴)) pero requiere función suave
   - Trapecio: Menor precisión (error O(h²)) pero más robusto

3. PASO ESPACIAL:
   - Se usa dy constante en el intervalo común de integración
   - 500 puntos de muestreo aseguran buena convergencia

4. RECOMENDACIÓN:
   La combinación más confiable depende de la suavidad del contorno.
   Para gotas bien formadas: Simpson + Spline
   Para contornos ruidosos: Trapecio + Spline o Simpson + Polinomio de grado moderado
""")

# Guardar resultados
df.to_csv('resultados_volumen_area.csv', index=False)
print("\nResultados guardados en 'resultados_volumen_area.csv'")