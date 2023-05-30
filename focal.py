# В зоне фокальной плоскости

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

def calculate_intensity(D, F, wavelength, x):
    k = 2 * np.pi / wavelength  # Волновое число

    # Расчет фазового фактора линзы
    def lens_phase(x, y):
        return np.exp(-1j * k * (x**2 + y**2) / (2 * F))

    # Расчет интенсивности на оси системы
    def intensity_at_axis(X, Y):
        r = np.sqrt(X**2 + Y**2)
        bessel_term = (2 * special.j1(k * r * r / (2 * F)) / (k * r * r / (2 * F)))**2
        lens_term = np.abs(lens_phase(X, Y))**2
        return bessel_term * lens_term

    X, Y = np.meshgrid(x, x)
    intensity = intensity_at_axis(X, Y)

    # Максимальная интенсивность и ее координата
    max_intensity_coord = np.unravel_index(np.argmax(intensity), intensity.shape)

    # Расчет энергии пучка по мере дифракции
    energy = np.sum(intensity) * (x[1] - x[0])**2

    return intensity, max_intensity_coord, energy

# Параметры системы
D = 1.0  # Диаметр отверстия
F = 10.0  # Фокусное расстояние линзы
wavelength = 0.5  # Длина волны
x = np.linspace(-10, 10, 500)  # Координаты на оси системы

# Расчет интенсивности, координаты максимальной интенсивности и энергии пучка
intensity, max_intensity_coord, energy = calculate_intensity(D, F, wavelength, x)

# Вывод координаты максимальной интенсивности
print("Координата максимальной интенсивности: ({}, {})".format(x[max_intensity_coord[1]], x[max_intensity_coord[0]]))

# График интенсивности в логарифмическом масштабе
plt.imshow(np.log10(intensity), cmap='hot', extent=[x[0], x[-1], x[0], x[-1]])
plt.colorbar(label='log10(Интенсивность)')
plt.plot(x[max_intensity_coord[1]], x[max_intensity_coord[0]], 'r+', label='Максимальная интенсивность')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Дифракционная картина')
plt.legend()
plt.show()

# Вывод величины энергии пучка
print("Энергия пучка: {}".format(energy))


