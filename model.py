import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rasterio
from scipy.ndimage import zoom

# Ruta al archivo .vrt (asegúrate de ajustar la ruta según tu ubicación)
vrt_path = "GTM_msk_alt.vrt"

# Valores mínimos y máximos de altitud que deseas visualizar (en metros)
altitud_min = -12000  # Puedes cambiar este valor
altitud_max = 5000  # Puedes cambiar este valor

# Factor de escala para reducir los picos de la superficie
factor_escala = 5 # Reducir aún más para evitar los picos

# Intentar abrir el archivo .vrt y leer los datos de elevación
try:
    with rasterio.open(vrt_path) as dataset:
        # Leer la banda de datos (asumimos que es una única banda con elevaciones)
        elevation = dataset.read(1)

        # Aplicar una máscara para excluir los valores fuera del rango
        elevation = np.where((elevation >= altitud_min) & (elevation <= altitud_max), elevation, np.nan)

        # Reducir la resolución para suavizar la visualización
        factor = 0.5  # Factor de reducción de la resolución (0.5 para más detalle, puedes ajustarlo)
        elevation_reduced = zoom(elevation, factor, order=1)

        # Rotar la matriz de elevación 90 grados para alinear la orientación
        elevation_rotated = np.rot90(elevation_reduced, k=3)  # Rotar 270 grados para corregir la orientación

        # Aplicar el factor de escala para comprimir los valores de altitud
        elevation_scaled = elevation_rotated * factor_escala

        # Aplicar una transformación logarítmica para mejorar la visualización
        elevation_scaled = np.log1p(elevation_scaled + 1)  # log1p para evitar log(0) y manejar valores pequeños

        # Obtener las nuevas dimensiones de la grilla reducida
        n_rows, n_cols = elevation_scaled.shape
        x = np.linspace(dataset.bounds.left, dataset.bounds.right, n_cols)
        y = np.linspace(dataset.bounds.top, dataset.bounds.bottom, n_rows)  # Invertir el orden de las coordenadas y
        x, y = np.meshgrid(x, y)

        # Crear una figura para visualizar en 3D
        fig = plt.figure(figsize=(15, 12))  # Aumentar el tamaño de la figura para mayor detalle
        ax = fig.add_subplot(111, projection='3d')

        # Graficar la superficie con los valores de altitud escalados
        ax.plot_surface(x, y, elevation_scaled, cmap='terrain', linewidth=0, antialiased=False)

        # Etiquetas y título
        ax.set_title("Visualización 3D Escalada y Transformada de la Superficie")
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.set_zlabel("Elevación (Escalada)")

        # Establecer límites para el eje Z para mejorar la visualización
        ax.set_zlim(0, 100)  # Comprimir los valores de Z a un rango más estrecho

        # Ajustar la vista inicial (ángulo de elevación y azimut)
        ax.view_init(elev=45, azim=180)  # Ajustar azimut a 180 grados para alinear correctamente la vista

        # Mostrar la gráfica
        plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"Error: {e}")
