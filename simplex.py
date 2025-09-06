import numpy as np

def print_tableau(table, headers):
    print("\n" + " | ".join(headers))
    print("-" * (len(" | ".join(headers))))
    for row in table:
        print(" | ".join([f"{x:.4f}" if isinstance(x, (float, np.float64)) else str(x) for x in row]))

def simplex_method(A, b, origin_var, art1_var, art2_var, num_equations):

    table = np.vstack([
        np.hstack([A, b.reshape(-1, 1)])  # Ограничения
    ])
    table = table.astype(np.float64)
    basis = [5, 3, 6]

    headers = [f"x{i+1}" for i in range(origin_var + art1_var + art2_var)] + ["b"]
    print_tableau(table, headers)
    
    # Этап 1: Минимизация w
    print("\nЭтап 1: Минимизация искусственной целевой функции w")
    while True:
        w_row = table[-1, :-1]
        min_value = np.min(w_row)
        min_index = np.argmin(w_row)  # Наибольший по модулю отрицательный коэффициент
        
        if min_value >= 0:
            break
        else:
            # Находим выводимую переменную (минимальное положительное отношение)
            ratios = np.array([])
            for i in range(num_equations):
                ratios = np.append(ratios, table[i, -1] / table[i, min_index])
            
            exiting = np.where(ratios >= 0)[0][np.argmin(ratios[ratios >= 0])] if np.any(ratios >= 0) else None
            
            # Обновляем базис
            del_elem = basis[exiting]
            basis[exiting] = min_index
            
            # Нормализуем строку выводимой переменной
            pivot = table[exiting, min_index]
            table[exiting, :] /= pivot
            
            # Обновляем остальные строки
            for i in range(num_equations+2):
                if i != exiting:
                    table[i, :] -= table[i, min_index] * table[exiting, :]
            
            print(f"\nИтерация: вводим x{min_index+1}, выводим x{del_elem + 1}")
            print_tableau(table, headers)
    
    
    # Этап 2: Удаляем искусственные переменные и решаем исходную задачу
    print("\nЭтап 2: Решение исходной задачи")
    
    # Удаляем строку w и искусственные переменные
    table = np.delete(table, -1, axis=0)  # Удаляем строку w
    table = np.delete(table, range(origin_var + art1_var, origin_var + art1_var + art2_var), axis=1)  # Удаляем искусственные переменные
    
    
    headers = [f"x{i+1}" for i in range(origin_var + art1_var)] + ["b"]
    print_tableau(table, headers)
    
    # Продолжаем симплекс-метод для исходной задачи
    while True:
        z_row = table[-1, :-1]

        min_value = np.min(z_row)
        min_index = np.argmin(z_row)  # Наибольший по модулю отрицательный коэффициент
        
        if min_value >= 0:
            break
        else:
            # Находим выводимую переменную (минимальное положительное отношение)
            ratios = np.array([])
            for i in range(num_equations):
                ratios = np.append(ratios, table[i, -1] / table[i, min_index])
            
            exiting = np.where(ratios >= 0)[0][np.argmin(ratios[ratios >= 0])] if np.any(ratios >= 0) else None

            
            # Обновляем базис
            del_elem = basis[exiting]
            basis[exiting] = min_index
            
            # Нормализуем строку выводимой переменной
            pivot = table[exiting, min_index]
            table[exiting, :] /= pivot
            
            # Обновляем остальные строки
            for i in range(num_equations + 1):
                if i != exiting:
                    table[i, :] -= table[i, min_index] * table[exiting, :]
            
            print(f"\nИтерация: вводим x{min_index+1}, выводим x{del_elem+1}")
            print_tableau(table, headers)
        
        
    
    # Извлекаем решение
    solution = np.zeros(origin_var)
    for i, var in enumerate(basis):
        if var < 2:
            solution[var] = table[i, -1]
    
    optimal_value = -table[-1, -1]
    
    return solution, optimal_value

""" 
Параметры задачи №2:
Минимизировать z = 3x1 + x2
При условиях:

x1 + 2x2 >= 2
x1 + x2 <= 3
2x1 + x2 >= 2
x1, x2 >= 0 

Введем искусственные переменные:

x1 + 2x2 - x3 + x6 = 2
x1 + x2 + x4 = 3
2x1 + x2 - x5 + x7 = 2

w = x6 + x7
w = 4 - 3x1 - 3x2 + x3 + x5

Составим матрицу коэффицентов A
И матрицу значений функций b
"""


A = np.array([
    [1, 2, -1, 0, 0, 1, 0],  # Коэффиценты первого уравнения
    [1, 1, 0, 1, 0, 0, 0],    # Коэффиценты второго уравнения
    [2, 1, 0, 0, -1, 0, 1], # Коэффиценты третьего уравнения
    [3, 1, 0, 0, 0, 0, 0], # Коэффиценты целевой функции z
    [-3, -3, 1, 0, 1, 0, 0]   # Коэффиценты искусственной целевой функции w
])

b = np.array([2, 3, 2, 0, -4]) # Матрица значений функций

origin_var = 2 # Количество исходных переменных
art1_var = 3 # Количество добавленных искусственных переменных
art2_var = 2 # Количество дополнительно добавленных искусственных переменных
num_equations = 3 # Количество уравнений в системе

# Применяем симплекс-метод
solution, optimal_value = simplex_method(A, b, origin_var, art1_var, art2_var, num_equations)

print("\nРезультат:")
print(f"Оптимальное решение: x1 = {solution[0]:.4f}, x2 = {solution[1]:.4f}")
print(f"Минимальное значение z = {optimal_value:.4f}")