from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from eda import EDAMaizArcorMejorado
from MLExtension import extender_eda_con_ml
import pandas as pd
from fastapi import Query
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from fastapi.responses import JSONResponse
from sklearn.metrics import mean_absolute_error, r2_score

app = FastAPI()

# CORS para permitir conexión desde el frontend React (puerto 5173 por ejemplo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",         # para desarrollo local fuera de Docker
    "http://frontend:5173",
    "*"      # para llamadas dentro de la red de Docker
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar análisis
dataset_paths = {
    'tipo_cambio': 'datos/tipos-de-cambio-historicos.csv',
    'maiz_datos': 'datos/maiz_ecc.csv',
    'ipc': 'datos/indice-precios-al-consumidor-nivel-general-base-diciembre-2016-mensual.csv',
    'clima': 'datos/Estadísticas normales Datos abiertos 1991-2020.xlsx',
    'agrofy_precios': 'datos/series-historicas-pizarra.csv'
}

eda = EDAMaizArcorMejorado(dataset_paths)
insights, df_maiz = eda.ejecutar_eda_completo_mejorado()
ml_extension, mejor_modelo = extender_eda_con_ml(eda)

@app.get("/api/predicciones/maiz")
def get_predicciones_maiz():
    df = ml_extension.df_ml.copy()
    modelo = ml_extension.models[mejor_modelo]['modelo']

    # Predecir sobre todo el dataset
    X = df[ml_extension.features].copy()
    X = X.fillna(0)

    y_pred = modelo.predict(X)

    df['prediccion'] = y_pred
    df = df.tail(24).reset_index()
    df['fecha'] = df['fecha'].astype(str)

    return df[['fecha', 'precio_maiz', 'prediccion']].rename(columns={'precio_maiz': 'precio'}).to_dict(orient="records")

@app.get("/api/insights")
def get_insights():
    def cast_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # convierte a int o float nativo
        elif isinstance(obj, dict):
            return {k: cast_numpy(v) for k, v in obj.items()}
        else:
            return obj

    return JSONResponse(content=cast_numpy(insights))


@app.get("/api/modelo")
def get_mejor_modelo():
    return {"modelo": mejor_modelo}

@app.get("/api/prediccion")
def predecir_precio_maiz(horizonte: int = Query(3, ge=1, le=12)):
    modelo = ml_extension.models[mejor_modelo]['modelo']
    df = ml_extension.df_ml.copy()
    ultimos = df.iloc[-1:].copy()
    predicciones = []
    fecha_actual = df.index[-1]

    for i in range(1, horizonte + 1):
        nueva_fecha = fecha_actual + relativedelta(months=i)
        row = ultimos.copy()
        row.index = [nueva_fecha]

        # Actualizar features temporales
        row['mes'] = nueva_fecha.month
        row['mes_sin'] = np.sin(2 * np.pi * nueva_fecha.month / 12)
        row['mes_cos'] = np.cos(2 * np.pi * nueva_fecha.month / 12)

        # Eliminar target
        row[ml_extension.target] = np.nan

        # Obtener predicción
        X_pred = row[ml_extension.features].fillna(0)  # ajustar según tu lógica real
        y_pred = modelo.predict(X_pred)[0]
        ultimos = row.copy()  # propaga valores para el siguiente paso

        predicciones.append({
            "mes": nueva_fecha.strftime("%b-%y"),
            "precio_predicho": round(float(y_pred), 2)
        })

    return {"horizonte": horizonte, "predicciones": predicciones}


@app.get("/api/escenarios")
def obtener_escenarios():
    """Devuelve precios simulados bajo escenarios hipotéticos"""
    base = ml_extension.df_ml["precio_maiz"].iloc[-1]

    escenarios = [
        { "escenario": "Base", "maiz": round(base), "probabilidad": 40 },
        { "escenario": "Devaluación 20%", "maiz": round(base * 1.2), "probabilidad": 20 },
        { "escenario": "Sequía Severa", "maiz": round(base * 1.18), "probabilidad": 15 },
        { "escenario": "Condiciones Óptimas", "maiz": round(base * 0.9), "probabilidad": 25 },
    ]

    return escenarios

@app.get("/api/historico/maiz")
def get_historico_maiz():
    df_maiz = ml_extension.df_ml.copy()
    df_maiz = df_maiz.reset_index()
    df_maiz['fecha'] = df_maiz['fecha'].astype(str)
    df_maiz = df_maiz.tail(24)
    
    return df_maiz[['fecha', 'precio_maiz']].to_dict(orient="records")

@app.get("/api/evaluacion_modelos")
def evaluacion_modelos():
    resultados = []
    for nombre, data in ml_extension.models.items():
        met = data.get("metricas_test", {})
        resultados.append({
            "nombre": nombre,
            "MAE": met.get("MAE"),
            "MAPE": met.get("MAPE"),
            "R2": met.get("R2"),
            "RMSE": met.get("RMSE"),
            "Precision_Direccion": met.get("Precision_Direccion")
        })
    return resultados

