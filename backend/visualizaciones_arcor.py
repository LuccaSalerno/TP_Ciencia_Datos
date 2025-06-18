
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def graficar_resultados(ml_pipeline):
    """
    Visualiza resultados de todos los modelos entrenados usando EnhancedRobustCornPipeline.
    """
    modelos = ml_pipeline.models
    if not modelos:
        print("❌ No hay modelos entrenados.")
        return

    y_test = list(modelos.values())[0]['y_test']
    fechas_test = ml_pipeline.df_ml.index[-len(y_test):]

    # Gráfico 1: Real vs Predicho
    plt.figure(figsize=(12, 6))
    plt.plot(fechas_test, y_test, label='Real', marker='o')
    for nombre, datos in modelos.items():
        plt.plot(fechas_test, datos['predictions_test'], label=nombre, marker='.')
    plt.title('📈 Predicción de Precio de Maíz - Test Set')
    plt.xticks(rotation=45)
    plt.ylabel('Precio (U$S)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Gráfico 2: Errores absolutos
    x = np.arange(len(fechas_test))
    bar_width = 0.8 / len(modelos)
    plt.figure(figsize=(12, 6))
    for i, (nombre, datos) in enumerate(modelos.items()):
        errores = np.abs(y_test - datos['predictions_test'])
        plt.bar(x + (i - len(modelos)/2) * bar_width, errores, width=bar_width, label=nombre)
    plt.xticks(x, [f.strftime('%b-%y') for f in fechas_test], rotation=45)
    plt.title('Error Absoluto por Modelo')
    plt.ylabel('Error (U$S)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Gráfico 3: MAPE Global por Modelo
    nombres = []
    mapes = []
    for nombre, datos in modelos.items():
        nombres.append(nombre)
        mapes.append(datos['test_mape'])
    plt.figure(figsize=(10, 6))
    plt.bar(nombres, mapes, color='skyblue')
    plt.title('MAPE Global por Modelo')
    plt.ylabel('MAPE (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Gráfico 4: Importancia de variables
    features = ml_pipeline.features
    plt.figure(figsize=(14, 6))
    plotted = False
    for nombre, datos in modelos.items():
        model = datos['model']
        if hasattr(model, 'feature_importances_'):
            importancia = model.feature_importances_
            if len(importancia) == len(features):
                plt.plot(features, importancia, marker='o', label=nombre)
                plotted = True
    if plotted:
        plt.title('Importancia de Variables por Modelo (solo modelos con feature_importances_)')
        plt.ylabel('Importancia Relativa')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ Ningún modelo tiene 'feature_importances_' para graficar.")
