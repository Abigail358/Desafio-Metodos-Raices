import math
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return x ** 3 - math.exp(0.8 * x) - 20
def df(x):
    return 3 * x ** 2 - 0.8 * math.exp(0.8 * x)


# MÉTODO DE BISECCIÓN
def metodo_biseccion(a, b, tol, max_iter=100):
    print("MÉTODO DE LA BISECCIÓN")
    print("=" * 80)
    print(f"Función: f(x) = x³ - e^(0.8x) - 20")
    print(f"Intervalo inicial: [{a}, {b}]")
    print(f"Tolerancia: {tol}")
    print()

    if f(a) * f(b) > 0:
        print("ERROR: No hay cambio de signo en el intervalo dado.")
        return None, []

    print(f"{'Iter':<4} {'a':<12} {'b':<12} {'m':<12} {'f(a)':<12} {'f(b)':<12} {'f(m)':<12} {'Error':<12}")
    print("-" * 100)
    iteracion = 0
    historial = []
    while iteracion < max_iter:
        m = (a + b) / 2
        fa = f(a)
        fb = f(b)
        fm = f(m)
        error = abs(b - a) / 2

        historial.append({
            'iter': iteracion, 'a': a, 'b': b, 'm': m,
            'fa': fa, 'fb': fb, 'fm': fm, 'error': error
        })

        print(f"{iteracion:<4} {a:<12.6f} {b:<12.6f} {m:<12.6f} {fa:<12.6f} {fb:<12.6f} {fm:<12.6f} {error:<12.6f}")

        if abs(fm) < tol:
            print(f"\n✓ CONVERGENCIA ALCANZADA - f(m) <= TOL")
            print(f"Raíz encontrada: x ≈ {m:.8f}")
            print(f"f({m:.8f}) = {fm:.10f}")
            return m, historial

        if fa * fm < 0:
            b = m  # La raíz está en [a, m]
        else:
            a = m  # La raíz está en [m, b]

        iteracion += 1

    print(f"\nMáximo de iteraciones alcanzado ({max_iter})")
    return m, historial


# MÉTODO DE NEWTON-RAPHSON
def metodo_newton_raphson(x0, tol, max_iter=100):
    print("\n" + "=" * 80)
    print("MÉTODO DE NEWTON-RAPHSON")
    print("=" * 80)
    print(f"Función: f(x) = x³ - e^(0.8x) - 20")
    print(f"Derivada: f'(x) = 3x² - 0.8e^(0.8x)")
    print(f"Valor inicial: x0 = {x0}")
    print(f"Tolerancia: {tol}")
    print()

    print(f"{'Iter':<4} {'x':<12} {'f(x)':<12} {'f\'(x)':<12} {'Error':<12}")
    print("-" * 60)

    iteracion = 0
    x_actual = x0
    historial = []

    while iteracion < max_iter:
        fx = f(x_actual)
        dfx = df(x_actual)
        if abs(dfx) < 1e-12:
            print("ERROR: Derivada cercana a cero.")
            return None, historial

        x_nuevo = x_actual - fx / dfx
        error = abs(x_nuevo - x_actual)

        historial.append({
            'iter': iteracion, 'x': x_actual, 'fx': fx,
            'dfx': dfx, 'error': error
        })

        print(f"{iteracion:<4} {x_actual:<12.6f} {fx:<12.6f} {dfx:<12.6f} {error:<12.6f}")

        if abs(fx) < tol:
            print(f"\n✓ CONVERGENCIA ALCANZADA - |f(x)| <= TOL")
            print(f"Raíz encontrada: x ≈ {x_actual:.8f}")
            print(f"f({x_actual:.8f}) = {fx:.10f}")
            return x_actual, historial

        x_actual = x_nuevo
        iteracion += 1

    print(f"\nMáximo de iteraciones alcanzado ({max_iter})")
    return x_actual, historial

# MÉTODO DE LA SECANTE
def metodo_secante(x0, x1, tol, max_iter=100):
    print("\n" + "=" * 80)
    print("MÉTODO DE LA SECANTE")
    print("=" * 80)
    print(f"Función: f(x) = x³ - e^(0.8x) - 20")
    print(f"Valores iniciales: x0 = {x0}, x1 = {x1}")
    print(f"Tolerancia: {tol}")
    print()
    print(f"{'Iter':<4} {'x':<12} {'f(x)':<12} {'Error':<12}")
    print("-" * 50)

    iteracion = 0
    x_prev = x0
    x_actual = x1
    historial = []

    while iteracion < max_iter:
        f_prev = f(x_prev)
        f_actual = f(x_actual)

        if abs(f_actual - f_prev) < 1e-12:
            print("ERROR: Diferencia de funciones cercana a cero.")
            return None, historial

        x_nuevo = x_actual - f_actual * (x_actual - x_prev) / (f_actual - f_prev)
        error = abs(f_actual)

        historial.append({
            'iter': iteracion, 'x': x_actual, 'fx': f_actual, 'error': error
        })

        print(f"{iteracion:<4} {x_actual:<12.6f} {f_actual:<12.6f} {error:<12.6f}")

        if abs(f_actual) < tol:
            print(f"\n✓ CONVERGENCIA ALCANZADA - |f(x)| <= TOL")
            print(f"Raíz encontrada: x ≈ {x_actual:.8f}")
            print(f"f({x_actual:.8f}) = {f_actual:.10f}")
            return x_actual, historial

        x_prev = x_actual
        x_actual = x_nuevo
        iteracion += 1

    print(f"\nMáximo de iteraciones alcanzado ({max_iter})")
    return x_actual, historial

# GRÁFICA
def graficar_funcion():
    plt.figure(figsize=(14, 10))
    def f_np(x):
        return x ** 3 - np.exp(0.8 * x) - 20
    x = np.linspace(-10, 10, 1000)
    y = f_np(x)
    plt.plot(x, y, 'b-', linewidth=2.5, label='f(x) = x³ - e^(0.8x) - 20')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    plt.xlim(-10, 10)
    plt.ylim(-30, 30)
    raices = []
    raiz1, _ = metodo_biseccion(2, 4, 1e-6)
    if raiz1 is not None:
        raices.append(raiz1)
        plt.plot(raiz1, 0, 'ro', markersize=10, label=f'Raíz A ≈ {raiz1:.2f}',
                 markeredgecolor='black', markeredgewidth=2)

    raiz2, _ = metodo_biseccion(7, 8, 1e-6)
    if raiz2 is not None:
        raices.append(raiz2)
        plt.plot(raiz2, 0, 'go', markersize=10, label=f'Raíz B ≈ {raiz2:.2f}',
                 markeredgecolor='black', markeredgewidth=2)

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('x', fontsize=12, fontweight='bold')
    plt.ylabel('f(x)', fontsize=12, fontweight='bold')
    plt.title('Gráfica de f(x) = x³ - e^(0.8x) - 20', fontsize=14, fontweight='bold')

    for i, raiz in enumerate(raices):
        plt.annotate(f'Raíz {chr(65 + i)}',
                     xy=(raiz, 0),
                     xytext=(raiz + 0.5, 5),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                     fontsize=10, fontweight='bold')

    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()

    print("\n" + "=" * 50)
    print("RAÍCES ENCONTRADAS:")
    print("=" * 50)
    for i, raiz in enumerate(raices):
        print(f"Raíz {chr(65 + i)}: x ≈ {raiz:.6f}, f(x) = {f(raiz):.2e}")

    plt.show()

    return raices

def encontrar_intervalo():
    print("BUSCANDO INTERVALO CON CAMBIO DE SIGNO")
    print("-" * 50)
    test_values = [2, 2.5, 3, 3.5, 4]

    for i in range(len(test_values) - 1):
        a, b = test_values[i], test_values[i + 1]
        fa, fb = f(a), f(b)

        print(f"Intervalo [{a:>3}, {b:>3}]: f({a:>3}) = {fa:>8.3f}, f({b:>3}) = {fb:>8.3f}", end="")

        if fa * fb < 0:
            print("  ✓ CAMBIO DE SIGNO")
            return a, b
        else:
            print("  ✗ mismo signo")

    return None, None


# PROGRAMA PRINCIPAL
def ejercicio_1():
    print("=" * 80)
    print("EJERCICIO 1 - ABIGAIL MAMANI")
    print("=" * 80)
    print("Función: f(x) = x³ - e^(0.8x) - 20")
    print()

    tolerancia = 1e-6

    a, b = encontrar_intervalo()
    print(f"\nUsando intervalo: [{a}, {b}]")

    # 1. MÉTODO DE BISECCIÓN
    raiz_biseccion, hist_biseccion = metodo_biseccion(a, b, tolerancia)

    # 2. MÉTODO DE NEWTON-RAPHSON
    x0_newton = 3.0
    raiz_newton, hist_newton = metodo_newton_raphson(x0_newton, tolerancia)

    # 3. MÉTODO DE LA SECANTE
    x0_secante, x1_secante = 3.0, 3.5
    raiz_secante, hist_secante = metodo_secante(x0_secante, x1_secante, tolerancia)

    # RESULTADOS COMPARATIVOS
    print("\n" + "=" * 80)
    print("RESULTADOS COMPARATIVOS")
    print("=" * 80)

    resultados = [
        ("Bisección", raiz_biseccion, len(hist_biseccion) if hist_biseccion else 0),
        ("Newton-Raphson", raiz_newton, len(hist_newton) if hist_newton else 0),
        ("Secante", raiz_secante, len(hist_secante) if hist_secante else 0)
    ]

    print(f"{'Método':<15} {'Raíz':<15} {'Iteraciones':<12} {'f(raíz)':<15}")
    print("-" * 60)

    for nombre, raiz, iteraciones in resultados:
        if raiz is not None:
            f_raiz = f(raiz)
            print(f"{nombre:<15} {raiz:<15.8f} {iteraciones:<12} {f_raiz:<15.10f}")
        else:
            print(f"{nombre:<15} {'No convergió':<15} {iteraciones:<12} {'-':<15}")

    # Generar gráfica
    print("\nGenerando gráfica.... calculando raices")
    print("\n")
    print("\n")
    print("\n")
    graficar_funcion()

    return resultados


# EJECUCIÓN

if __name__ == "__main__":
    ejercicio_1()