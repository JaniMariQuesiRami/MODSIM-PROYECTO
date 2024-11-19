# Simulación de Ondas de Expansión en Guatemala

Este proyecto simula cómo una onda de expansión, causada por una explosión, se propaga sobre la topografía real de Guatemala. Utiliza un modelo de **difusión** y datos topográficos reales para representar el terreno. Se aplicaron diferentes **factores de escala** para ajustar los parámetros y obtener una visualización más realista.

## Demostración

![Demostración de la Simulación](ruta/al/archivo.gif)

## Descripción

- **Modelo de Difusión**: Se implementa la ecuación de difusión para simular la propagación de la onda de expansión.
- **Topografía Real**: Se utilizan datos de elevación reales de Guatemala para representar el terreno en la simulación.
- **Factores de Escala**: Se ajustan los parámetros de la simulación, como el coeficiente de difusión y el tamaño de la malla, para lograr resultados coherentes y manejables computacionalmente.

## Requisitos

- **Python 3.x**
- Bibliotecas de Python:
  - numpy
  - matplotlib
  - rasterio
  - scipy
  - numba
  - tkinter (generalmente incluida con Python)

## Instrucciones

1. **Clonar el repositorio**:

   ```bash
   git clone https://github.com/JaniMariQuesiRami/MODSIM-PROYECTO.git

2. **Instalar las dependencias**:

   ```bash
   pip install numpy matplotlib rasterio scipy numba

3. **Ejecutar la simulación**:

   ```bash
   python simulación.py

