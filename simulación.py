import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk
import rasterio
from scipy.ndimage import zoom
from numba import jit
import tkinter.font as tkFont  # Para gestionar fuentes en Tkinter

# Parámetros de simulación
Lx, Ly = 100, 100  # Tamaño de la malla para la onda de expansión (200x200 puntos)
dx = dy = 1        # Tamaño del paso espacial en ambas direcciones (unidades arbitrarias)
D = 1.0            # Coeficiente de difusión (controla la rapidez de propagación)
dt = 0.15           # Tamaño del paso temporal (unidades arbitrarias)

# Opciones de tamaño de explosión y tiempo adicional
# Diccionario que asocia tipos de explosiones con su intensidad inicial y tiempo de simulación
explosiones = {
    "Little Boy (Hiroshima)(15kt)": (1500, 20),    # ~15 kilotones, ajustado a 20 segundos
    "Castle Bravo(15k kt)": (150000, 120),         # ~15,000 kilotones, ajustado a 2 minutos
    "Fat Man (Nagasaki)(21 kt)": (2100, 30),       # ~21 kilotones, ajustado a 30 segundos
    "Tsar Bomba(50k kt)": (500000, 300)            # ~50,000 kilotones, ajustado a 5 minutos
}

# Coordenadas predefinidas para los 22 departamentos de Guatemala
# Cada departamento tiene asignadas coordenadas (x, y) en la malla de simulación
departamentos_coordenadas = {
    "Guatemala": (25, 25),
    "Huehuetenango": (10, 10),
    "Quiché": (15, 12),
    "Alta Verapaz": (18, 20),
    "Baja Verapaz": (22, 18),
    "Chimaltenango": (24, 22),
    "Chiquimula": (30, 35),
    "El Progreso": (27, 24),
    "Escuintla": (35, 20),
    "Izabal": (32, 40),
    "Jalapa": (28, 30),
    "Jutiapa": (34, 28),
    "Petén": (5, 40),
    "Quetzaltenango": (12, 15),
    "Retalhuleu": (15, 8),
    "Sacatepéquez": (26, 23),
    "San Marcos": (10, 5),
    "Santa Rosa": (30, 20),
    "Sololá": (18, 15),
    "Suchitepéquez": (20, 10),
    "Totonicapán": (14, 14),
    "Zacapa": (28, 38),
    "Coordenadas Personalizadas": (0, 0)  # Permite al usuario ingresar coordenadas manualmente
}

@jit(nopython=True)
def diffusion_step(u, D, dx, dy, dt):
    """
    Realiza un paso temporal de la ecuación de difusión en 2D.

    Parámetros:
    - u: matriz 2D de la variable dependiente (intensidad de la onda)
    - D: coeficiente de difusión
    - dx, dy: pasos espaciales en las direcciones x e y
    - dt: paso temporal

    Retorna:
    - new_u: matriz 2D actualizada después del paso de difusión
    """
    new_u = u.copy()  # Crear una copia para almacenar los nuevos valores sin modificar los actuales
    for i in range(1, Lx-1):
        for j in range(1, Ly-1):
            # Aplicar la ecuación de diferencias finitas para la difusión en 2D
            new_u[i, j] = u[i, j] + D * dt * (
                (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 +
                (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            )
    return new_u

# Leer el archivo .vrt y preparar los datos de elevación
try:
    with rasterio.open("GTM_msk_alt.vrt") as dataset:
        elevation = dataset.read(1)  # Leer la banda de datos de elevación
        altitud_min = -12000
        altitud_max = 5000
        # Filtrar valores de elevación fuera del rango especificado
        elevation = np.where((elevation >= altitud_min) & (elevation <= altitud_max), elevation, np.nan)
        factor = 0.5  # Factor para reducir la resolución (ajuste computacional)
        elevation_reduced = zoom(elevation, factor, order=1)  # Reducir resolución de la matriz de elevación
        elevation_rotated = np.rot90(elevation_reduced, k=3)  # Rotar la matriz para que coincida con la orientación deseada
        factor_escala = 5  # Factor para escalar las alturas y mejorar la visualización
        elevation_scaled = elevation_rotated * factor_escala  # Escalar los valores de elevación
        elevation_scaled = np.log1p(elevation_scaled + 1)  # Aplicar logaritmo para realzar diferencias en altitudes bajas
        n_rows, n_cols = elevation_scaled.shape  # Obtener dimensiones de la matriz de elevación
        # Crear mallas de coordenadas X e Y basadas en los límites geográficos del dataset
        x = np.linspace(dataset.bounds.left, dataset.bounds.right, n_cols)
        y = np.linspace(dataset.bounds.top, dataset.bounds.bottom, n_rows)
        x, y = np.meshgrid(x, y)  # Crear malla de coordenadas
except Exception as e:
    print(f"Error al cargar el archivo de elevación: {e}")
    # Si ocurre un error, inicializar elevación plana y mallas X, Y básicas
    elevation_scaled = np.zeros((Lx, Ly))
    x, y = np.meshgrid(np.arange(Lx), np.arange(Ly))

# Clase para la aplicación gráfica utilizando Tkinter
class ExplosionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulación de Ondas de Expansión")  # Título de la ventana principal
        self.geometry("2000x2000")  # Dimensiones de la ventana
        # Variables para los elementos interactivos de la interfaz
        self.tipo_explosion_var = tk.StringVar(value="Little Boy (Hiroshima)(15kt)")
        self.departamento_var = tk.StringVar(value="Guatemala")
        self.coord_x_var = tk.IntVar(value=departamentos_coordenadas["Guatemala"][0])
        self.coord_y_var = tk.IntVar(value=departamentos_coordenadas["Guatemala"][1])
        self.simulando = False  # Indicador para controlar el estado de la simulación
        self.create_widgets()  # Llamada al método para crear los widgets de la interfaz

    def create_widgets(self):
        # Crear el frame para los controles (panel lateral)
        control_frame = tk.Frame(self, width=400)  # Ancho ajustado para acomodar los elementos
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Etiqueta y menú desplegable para seleccionar el tipo de explosión
        tk.Label(control_frame, text="Tipo de Explosión:", font=("Arial", 14)).pack(pady=10)
        menu_explosion = tk.OptionMenu(control_frame, self.tipo_explosion_var, *explosiones.keys())
        menu_explosion.config(font=("Arial", 14))  # Establecer fuente del menú
        menu_explosion["menu"].config(font=("Arial", 14))  # Establecer fuente de las opciones
        menu_explosion.pack(pady=15)

        # Etiqueta y menú desplegable para seleccionar el departamento
        tk.Label(control_frame, text="Departamento:", font=("Arial", 14)).pack(pady=10)
        menu_departamento = tk.OptionMenu(control_frame, self.departamento_var, *departamentos_coordenadas.keys(), command=self.update_coords)
        menu_departamento.config(font=("Arial", 14))
        menu_departamento["menu"].config(font=("Arial", 14))
        menu_departamento.pack(pady=15, padx=5)

        # Frame para las entradas de coordenadas X e Y
        coord_frame = tk.Frame(control_frame)
        coord_frame.pack(pady=10)

        # Etiqueta y entrada para la coordenada X
        ttk.Label(coord_frame, text="Coordenada X:", font=("Arial", 14)).grid(row=0, column=0, padx=5, pady=5)
        self.coord_x_entry = ttk.Entry(coord_frame, textvariable=self.coord_x_var, width=10, font=("Arial", 14))
        self.coord_x_entry.grid(row=0, column=1, padx=5, pady=5)

        # Etiqueta y entrada para la coordenada Y
        ttk.Label(coord_frame, text="Coordenada Y:", font=("Arial", 14)).grid(row=1, column=0, padx=5, pady=5)
        self.coord_y_entry = ttk.Entry(coord_frame, textvariable=self.coord_y_var, width=10, font=("Arial", 14))
        self.coord_y_entry.grid(row=1, column=1, padx=5, pady=5)

        # Botón para iniciar la simulación
        ttk.Button(control_frame, text="Iniciar Simulación", command=self.run_simulation, style="TButton").pack(pady=20)

        # Estilo personalizado para el botón
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 14), padding=10)

        # Área para el gráfico 3D de la simulación
        self.fig = plt.figure(figsize=(15, 12))  # Tamaño de la figura de Matplotlib
        self.ax = self.fig.add_subplot(111, projection='3d')  # Añadir un subplot en 3D
        self.canvas = FigureCanvasTkAgg(self.fig, self)  # Integrar la figura en el canvas de Tkinter
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # Ubicar el canvas en la interfaz

    def update_coords(self, *args):
        """
        Actualiza las coordenadas cuando se selecciona un departamento.
        Si el usuario elige "Coordenadas Personalizadas", habilita las entradas para que el usuario pueda ingresar valores.
        """
        departamento = self.departamento_var.get()
        if departamento == "Coordenadas Personalizadas":
            # Habilitar los campos de entrada de coordenadas
            self.coord_x_entry.config(state='normal')
            self.coord_y_entry.config(state='normal')
            self.coord_x_var.set(0)  # Restablecer a 0 para entradas personalizadas
            self.coord_y_var.set(0)
        else:
            # Deshabilitar las entradas y asignar las coordenadas predefinidas del departamento seleccionado
            coords = departamentos_coordenadas.get(departamento, (0, 0))
            self.coord_x_var.set(coords[0])
            self.coord_y_var.set(coords[1])
            self.coord_x_entry.config(state='disabled')
            self.coord_y_entry.config(state='disabled')

    def run_simulation(self):
        """
        Inicia y ejecuta la simulación de la onda de expansión basada en los parámetros seleccionados.
        """
        tipo_explosion = self.tipo_explosion_var.get()
        intensidad_inicial, T = explosiones[tipo_explosion]  # Obtener intensidad y tiempo de la explosión seleccionada
        x_explosion = self.coord_x_var.get()
        y_explosion = self.coord_y_var.get()

        # Crear la matriz inicial de la onda, con la intensidad inicial en el punto de la explosión
        u = np.zeros((Lx, Ly))
        u[x_explosion, y_explosion] = intensidad_inicial

        # Ajustar las mallas de coordenadas y la elevación al tamaño de la simulación
        X_resized = zoom(x, (Lx / n_rows, Ly / n_cols), order=1)
        Y_resized = zoom(y, (Lx / n_rows, Ly / n_cols), order=1)
        Z_resized = zoom(elevation_scaled, (Lx / n_rows, Ly / n_cols), order=1)

        self.simulando = True  # Establecer bandera para indicar que la simulación está en curso

        # Bucle principal de la simulación, iterando sobre los pasos de tiempo
        for t in range(int(T / dt)):
            if not self.simulando:
                break  # Si la simulación ha sido detenida, salir del bucle

            u = diffusion_step(u, D, dx, dy, dt)  # Realizar un paso de la ecuación de difusión

            # Limpiar el gráfico previo para actualizar con los nuevos datos
            self.ax.clear()
            # Graficar la superficie del terreno
            self.ax.plot_surface(X_resized, Y_resized, Z_resized, cmap='terrain', alpha=0.7)
            self.ax.view_init(elev=40, azim=170)  # Ajustar ángulo de visión para mejor visualización

            # Calcular la superficie de la onda de expansión sobre el terreno
            explosion_surface = Z_resized + (u / u.max()) * 0.5  # Escalar la intensidad de la onda para visualizarla

            # Graficar la onda de expansión con un mapa de colores que representa la intensidad
            self.ax.plot_surface(
                X_resized, Y_resized, explosion_surface, facecolors=plt.cm.hot(u / u.max()),
                shade=False, alpha=0.6
            )
            self.ax.set_zlim(-10, 50)  # Establecer límites en el eje Z para mantener una escala consistente
            self.ax.set_title(f"Paso de Tiempo {t}, Tipo: {tipo_explosion}")  # Actualizar el título con información actual
            self.canvas.draw()  # Dibujar el nuevo frame en el canvas
            self.update()  # Actualizar la interfaz de Tkinter

    def on_close(self):
        """
        Maneja el evento de cierre de la ventana.
        Detiene la simulación y cierra los recursos asociados.
        """
        self.simulando = False  # Detener la simulación si está en curso
        self.destroy()  # Cerrar la ventana de la aplicación
        plt.close('all')  # Cerrar todas las figuras de Matplotlib
        self.quit()  # Salir de la aplicación de Tkinter

if __name__ == "__main__":
    app = ExplosionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)  # Asociar la función on_close al evento de cerrar la ventana
    app.mainloop()  # Iniciar el bucle principal de la aplicación
