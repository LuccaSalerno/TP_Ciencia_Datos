import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def graficar_resultados(ml_ext):
    """
    Genera visualizaciones automáticas a partir del objeto MLExtensionMaizArcor ya entrenado.
    """
    resultados = ml_ext.models
    fechas = ml_ext.test_index

    # Obtener y_test y predicciones reales de cada modelo
    y_test = ml_ext.df_ml.loc[fechas][ml_ext.target]
    pred_rf = resultados['Random Forest']['predicciones_test']
    pred_xgb = resultados['XGBoost']['predicciones_test']
    pred_lr = resultados['Linear Regression']['predicciones_test']

    # Errores absolutos
    err_rf = np.abs(y_test - pred_rf)
    err_xgb = np.abs(y_test - pred_xgb)
    err_lr = np.abs(y_test - pred_lr)

    # Importancia de variables
    try:
        rf_importance = resultados['Random Forest']['modelo'].feature_importances_
        xgb_importance = resultados['XGBoost']['modelo'].feature_importances_
        features = ml_ext.features
    except:
        rf_importance = xgb_importance = features = []

    # Gráfico 1: Real vs Predicho
    plt.figure(figsize=(10, 5))
    plt.plot(fechas, y_test, label='Real', marker='o')
    plt.plot(fechas, pred_rf, label='Random Forest', marker='s')
    plt.plot(fechas, pred_xgb, label='XGBoost', marker='^')
    plt.plot(fechas, pred_lr, label='Linear Regression', marker='x')
    plt.title('📈 Predicción de Precio de Maíz - Test Set')
    plt.xticks(rotation=45)
    plt.ylabel('Precio (U$S)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Gráfico 2: Errores absolutos
    x = np.arange(len(fechas))
    bar_width = 0.25
    plt.figure(figsize=(10, 5))
    plt.bar(x - bar_width, err_rf, width=bar_width, label='Random Forest')
    plt.bar(x, err_xgb, width=bar_width, label='XGBoost')
    plt.bar(x + bar_width, err_lr, width=bar_width, label='Linear Regression')
    plt.xticks(x, [f.strftime('%b-%y') for f in fechas], rotation=45)
    plt.title('Error Absoluto por Modelo')
    plt.ylabel('Error (U$S)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Gráfico 3: MAPE Global
    modelos = ['Linear Regression', 'Random Forest', 'XGBoost']
    mape = [
        resultados['Linear Regression']['metricas_test']['MAPE'],
        resultados['Random Forest']['metricas_test']['MAPE'],
        resultados['XGBoost']['metricas_test']['MAPE']
    ]
    plt.figure(figsize=(8, 5))
    plt.bar(modelos, mape, color=['gray', 'green', 'red'])
    plt.title('MAPE Global por Modelo')
    plt.ylabel('MAPE (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Gráfico 4: Importancia de variables
    if len(features) > 0:
        x = np.arange(len(features))
        bar_width = 0.35
        plt.figure(figsize=(12, 6))
        plt.bar(x - bar_width/2, rf_importance, width=bar_width, label='Random Forest')
        plt.bar(x + bar_width/2, xgb_importance, width=bar_width, label='XGBoost')
        plt.xticks(x, features, rotation=45, ha='right')
        plt.title('Importancia de Variables')
        plt.ylabel('Importancia Relativa')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ No se pudo calcular la importancia de variables para los modelos.")
