from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from eda import EDAMaizArcorMejorado
import pandas as pd
from fastapi import Query
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from fastapi.responses import JSONResponse
from sklearn.metrics import mean_absolute_error, r2_score
from MLExtension import EnhancedRobustCornPipeline  # Importamos la nueva clase
from eda import EDAMaizArcorMejorado

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

dataset_paths = {
    'tipo_cambio': 'datos/tipos-de-cambio-historicos.csv',
    'maiz_datos': 'datos/maiz_ecc.csv',
    'ipc': 'datos/indice-precios-al-consumidor-nivel-general-base-diciembre-2016-mensual.csv',
    'clima': 'datos/Estadísticas normales Datos abiertos 1991-2020.xlsx',
    'agrofy_precios': 'datos/series-historicas-pizarra.csv'
}

# Inicializar EDA
eda = EDAMaizArcorMejorado(dataset_paths)
insights, df_maiz = eda.ejecutar_eda_completo_mejorado()

# Inicializar Enhanced ML Pipeline
ml_pipeline = EnhancedRobustCornPipeline(eda)

# Ejecutar análisis completo
print("🚀 Ejecutando análisis ML mejorado...")
ml_results = ml_pipeline.run_complete_analysis()

# Identificar el mejor modelo
def get_best_model_name():
    """Obtiene el nombre del mejor modelo basado en métricas"""
    if not ml_pipeline.models:
        return None
    
    # Filtrar modelos con R² positivo
    good_models = {k: v for k, v in ml_pipeline.models.items() if v['test_r2'] > 0}
    
    if good_models:
        # Mejor modelo por MAPE entre los que tienen R² > 0
        return min(good_models.keys(), key=lambda x: good_models[x]['test_mape'])
    else:
        # Si ninguno tiene R² positivo, usar el de menor MAPE
        return min(ml_pipeline.models.keys(), key=lambda x: ml_pipeline.models[x]['test_mape'])

mejor_modelo = get_best_model_name()

@app.get("/api/predicciones/maiz")
def get_predicciones_maiz():
    """Obtiene predicciones vs valores reales para los últimos 24 meses"""
    if not ml_pipeline.df_ml is not None and mejor_modelo:
        return {"error": "ML pipeline not initialized"}
    
    df = ml_pipeline.df_ml.copy()
    modelo_data = ml_pipeline.models[mejor_modelo]
    modelo = modelo_data['model']
    
    # Preparar datos para predicción
    X = df[ml_pipeline.features].copy()
    X = ml_pipeline.clean_infinity_values(X)
    X = X.fillna(0)
    
    # Aplicar escalado si es necesario
    if modelo_data['config']['use_scaling']:
        scaler = ml_pipeline.scalers['robust']
        X_scaled = scaler.transform(X)
        y_pred = modelo.predict(X_scaled)
    else:
        y_pred = modelo.predict(X.values)
    
    df['prediccion'] = y_pred
    df = df.tail(60).reset_index()
    df['fecha'] = df['fecha'].astype(str)
    
    return df[['fecha', 'precio_maiz', 'prediccion']].rename(
        columns={'precio_maiz': 'precio'}
    ).to_dict(orient="records")

@app.get("/api/insights")
def get_insights():
    """Obtiene insights del EDA con conversión de tipos numpy"""
    def cast_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: cast_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [cast_numpy(item) for item in obj]
        else:
            return obj
    
    # Combinar insights del EDA con insights del ML
    enhanced_insights = insights.copy()
    
    if ml_pipeline.models:
        enhanced_insights['ml_performance'] = {
            'best_model': mejor_modelo,
            'models_trained': len(ml_pipeline.models),
            'features_used': len(ml_pipeline.features),
            'dataset_size': len(ml_pipeline.df_ml) if ml_pipeline.df_ml is not None else 0
        }
        
        # Añadir métricas del mejor modelo
        if mejor_modelo and mejor_modelo in ml_pipeline.models:
            best_metrics = ml_pipeline.models[mejor_modelo]
            enhanced_insights['ml_performance']['best_model_metrics'] = {
                'test_mape': best_metrics['test_mape'],
                'test_r2': best_metrics['test_r2'],
                'test_rmse': best_metrics['test_rmse'],
                'test_mae': best_metrics['test_mae']
            }
    
    return JSONResponse(content=cast_numpy(enhanced_insights))

@app.get("/api/modelo")
def get_mejor_modelo():
    """Obtiene información del mejor modelo"""
    if not mejor_modelo:
        return {"error": "No model available"}
    
    model_info = {
        "modelo": mejor_modelo,
        "metricas": ml_pipeline.models[mejor_modelo] if mejor_modelo in ml_pipeline.models else None
    }
    
    # Convertir numpy types
    if model_info["metricas"]:
        metrics = model_info["metricas"]
        model_info["metricas_clean"] = {
            "test_mape": float(metrics['test_mape']),
            "test_r2": float(metrics['test_r2']),
            "test_rmse": float(metrics['test_rmse']),
            "test_mae": float(metrics['test_mae']),
            "pred_std": float(metrics['pred_std'])
        }
    
    return model_info

@app.get("/api/prediccion")
def predecir_precio_maiz(horizonte: int = Query(3, ge=1, le=12)):
    """Predicción futura con recalculo completo de features"""
    if ml_pipeline.df_ml is None or not mejor_modelo:
        return {"error": "ML pipeline not initialized"}
    
    modelo_data = ml_pipeline.models[mejor_modelo]
    modelo = modelo_data['model']
    pred_std = modelo_data['pred_std']
    df_base = ml_pipeline.df_ml.copy()

    # Inicializar con los últimos datos reales
    df_simulado = df_base.copy()
    fecha_actual = df_base.index[-1]
    predicciones = []

    for i in range(1, horizonte + 1):
        nueva_fecha = fecha_actual + relativedelta(months=1)

        # Crear un nuevo registro con fecha futura
        nueva_fila = df_simulado.iloc[[-1]].copy()
        nueva_fila.index = [nueva_fecha]

        # Predecir usando los features actualizados
        df_simulado = pd.concat([df_simulado, nueva_fila])

        # Recalcular todos los features
        df_features = ml_pipeline.engineer_enhanced_features(df_simulado)

        # Usar solo la última fila para la predicción actual
        row = df_features.iloc[[-1]].copy()
        row = ml_pipeline.clean_infinity_values(row)

        # Rellenar faltantes con medianas del entrenamiento
        if hasattr(ml_pipeline, "feature_medians"):
            row = row.fillna(ml_pipeline.feature_medians)
        else:
            row = row.fillna(method="ffill").fillna(method="bfill")

        # Predicción
        X_pred = row[ml_pipeline.features]
        if modelo_data['config']['use_scaling']:
            scaler = ml_pipeline.scalers['robust']
            X_pred_scaled = scaler.transform(X_pred)
            y_pred = modelo.predict(X_pred_scaled)[0]
        else:
            y_pred = modelo.predict(X_pred.values)[0]

        # Calcular intervalos de confianza
        confidence_factor = 1.96
        uncertainty_growth = np.sqrt(i)
        margin_error = confidence_factor * pred_std * uncertainty_growth

        predicciones.append({
            "mes": nueva_fecha.strftime("%b-%y"),
            "precio_predicho": round(float(y_pred), 2),
            "limite_inferior": round(float(y_pred - margin_error), 2),
            "limite_superior": round(float(y_pred + margin_error), 2),
            "confianza": round(100 / uncertainty_growth, 1)
        })

        # Actualizar el dataframe con el nuevo valor predicho
        df_simulado.at[nueva_fecha, ml_pipeline.target] = y_pred
        fecha_actual = nueva_fecha

    return {
        "horizonte": horizonte,
        "modelo_usado": mejor_modelo,
        "predicciones": predicciones
    }



@app.get("/api/escenarios")
def obtener_escenarios():
    """Escenarios simulados con el modelo ML"""
    if ml_pipeline.df_ml is None or not mejor_modelo:
        return {"error": "ML pipeline not initialized"}
    
    modelo_data = ml_pipeline.models[mejor_modelo]
    modelo = modelo_data['model']
    pred_std = modelo_data['pred_std']
    base_df = ml_pipeline.df_ml.copy()
    ultima_fila = base_df.iloc[[-1]].copy()
    fecha_pred = base_df.index[-1] + relativedelta(months=1)

    escenarios_definidos = [
        {
            "escenario": "Base",
            "modificaciones": {},
            "probabilidad": 40,
            "descripcion": "Condiciones actuales se mantienen"
        },
        {
            "escenario": "Devaluación 20%",
            "modificaciones": {
                "tipo_cambio": ultima_fila["tipo_cambio"].values[0] * 1.2
            },
            "probabilidad": 20,
            "descripcion": "Devaluación significativa del peso"
        },
        {
            "escenario": "Sequía Severa",
            "modificaciones": {
                "precio_maiz": ultima_fila["precio_maiz"].values[0] * 1.15,
                "harvest_season": 1
            },
            "probabilidad": 15,
            "descripcion": "Condiciones climáticas adversas"
        },
        {
            "escenario": "Condiciones Óptimas",
            "modificaciones": {
                "precio_maiz": ultima_fila["precio_maiz"].values[0] * 0.85,
                "harvest_season": 1
            },
            "probabilidad": 25,
            "descripcion": "Cosecha abundante y condiciones favorables"
        },
        {
            "escenario": "Intervención Estatal",
            "modificaciones": {
                "precio_maiz": ultima_fila["precio_maiz"].values[0] * 0.8,
                "tc_volatility": 0.0
            },
            "probabilidad": 10,
            "descripcion": "Intervención en el mercado local de granos que limita exportaciones y afecta precios internos"
        }
    ]

    resultados = []

    for escenario in escenarios_definidos:
        df_simulado = base_df.copy()
        nueva_fila = ultima_fila.copy()
        nueva_fila.index = [fecha_pred]

        # Aplicar modificaciones del escenario
        for k, v in escenario["modificaciones"].items():
            nueva_fila[k] = v

        df_simulado = pd.concat([df_simulado, nueva_fila])
        df_features = ml_pipeline.engineer_enhanced_features(df_simulado)
        row = df_features.iloc[[-1]].copy()
        row = ml_pipeline.clean_infinity_values(row)

        if hasattr(ml_pipeline, "feature_medians"):
            row = row.fillna(ml_pipeline.feature_medians)
        else:
            row = row.fillna(method='ffill').fillna(method='bfill')

        X_pred = row[ml_pipeline.features]
        if modelo_data['config']['use_scaling']:
            scaler = ml_pipeline.scalers['robust']
            X_pred = scaler.transform(X_pred)

        y_pred = modelo.predict(X_pred)[0]
        margin_error = 1.96 * pred_std

        resultados.append({
            "escenario": escenario["escenario"],
            "maiz": round(float(y_pred), 2),
            "limite_inferior": round(float(y_pred - margin_error), 2),
            "limite_superior": round(float(y_pred + margin_error), 2),
            "probabilidad": escenario["probabilidad"],
            "descripcion": escenario["descripcion"]
        })

    volatilidad = base_df["precio_maiz"].pct_change().std() * 100
    precio_base = base_df["precio_maiz"].iloc[-1]

    return {
        "escenarios": resultados,
        "volatilidad_historica": round(volatilidad, 2),
        "precio_base": round(precio_base, 2)
    }


@app.get("/api/historico/maiz")
def get_historico_maiz():
    """Obtiene datos históricos de maíz"""
    if not ml_pipeline.df_ml is not None:
        return {"error": "ML pipeline not initialized"}
    
    df_maiz = ml_pipeline.df_ml.copy()
    df_maiz = df_maiz.reset_index()
    df_maiz['fecha'] = df_maiz['fecha'].astype(str)
    df_maiz = df_maiz.tail(60)
    
    return df_maiz[['fecha', 'precio_maiz']].to_dict(orient="records")

@app.get("/api/evaluacion_modelos")
def evaluacion_modelos():
    """Evaluación detallada de todos los modelos entrenados"""
    if not ml_pipeline.models:
        return {"error": "No models trained"}
    
    resultados = []
    for nombre, data in ml_pipeline.models.items():
        resultado = {
            "nombre": nombre,
            "MAE": round(float(data["test_mae"]), 2),
            "MAPE": round(float(data["test_mape"]), 2),
            "R2": round(float(data["test_r2"]), 3),
            "RMSE": round(float(data["test_rmse"]), 2),
            "pred_std": round(float(data["pred_std"]), 2),
            "usa_escalado": data["config"]["use_scaling"],
            "es_mejor": (nombre == mejor_modelo)
        }
        resultados.append(resultado)
    
    # Ordenar por MAPE
    resultados.sort(key=lambda x: x["MAPE"])
    
    return {
        "modelos": resultados,
        "mejor_modelo": mejor_modelo,
        "total_features": len(ml_pipeline.features) if ml_pipeline.features else 0
    }

@app.get("/api/feature_importance")
def get_feature_importance():
    """Obtiene la importancia de las features del mejor modelo"""
    if not mejor_modelo or mejor_modelo not in ml_pipeline.feature_importance:
        return {"error": "Feature importance not available"}
    
    importance = ml_pipeline.feature_importance[mejor_modelo]
    
    # Convertir a formato API
    features_data = []
    for feature, valor in importance.head(15).items():  # Top 15 features
        features_data.append({
            "feature": feature,
            "importancia": round(float(valor), 4),
            "importancia_pct": round(float(valor / importance.sum() * 100), 2)
        })
    
    return {
        "modelo": mejor_modelo,
        "features": features_data,
        "total_features": len(ml_pipeline.features)
    }

@app.get("/api/diagnosticos")
def get_diagnosticos():
    """Diagnósticos del modelo y calidad de datos"""
    if not ml_pipeline.df_ml is not None or not mejor_modelo:
        return {"error": "ML pipeline not initialized"}
    
    df = ml_pipeline.df_ml
    modelo_data = ml_pipeline.models[mejor_modelo]
    
    diagnosticos = {
        "datos": {
            "total_observaciones": len(df),
            "rango_fechas": {
                "inicio": df.index.min().strftime("%Y-%m-%d"),
                "fin": df.index.max().strftime("%Y-%m-%d")
            },
            "valores_faltantes": int(df.isnull().sum().sum()),
            "precio_promedio": round(float(df['precio_maiz'].mean()), 2),
            "precio_volatilidad": round(float(df['precio_maiz'].std()), 2)
        },
        "modelo": {
            "nombre": mejor_modelo,
            "precision": round(float(modelo_data['test_mape']), 2),
            "r_cuadrado": round(float(modelo_data['test_r2']), 3),
            "error_promedio": round(float(modelo_data['test_mae']), 2),
            "features_utilizadas": len(ml_pipeline.features),
            "usa_escalado": modelo_data['config']['use_scaling']
        },
        "rendimiento": {
            "predicciones_test": len(modelo_data['y_test']),
            "error_std": round(float(modelo_data['pred_std']), 2),
            "sesgo_promedio": round(float(np.mean(modelo_data['residuals'])), 2)
        }
    }
    
    return diagnosticos

# Endpoint de salud
@app.get("/api/health")
def health_check():
    """Health check del API"""
    status = {
        "status": "healthy",
        "eda_initialized": eda is not None,
        "ml_pipeline_initialized": ml_pipeline is not None,
        "models_trained": len(ml_pipeline.models) if ml_pipeline.models else 0,
        "best_model": mejor_modelo,
        "timestamp": datetime.now().isoformat()
    }
    return status