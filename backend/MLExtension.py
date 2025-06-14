import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

import warnings
warnings.filterwarnings('ignore')

class MLExtensionMaizArcor:
    """Extensión ML para el EDA de Maíz Arcor"""
    
    def __init__(self, eda_instance):
        self.eda = eda_instance
        self.df_ml = None
        self.features = None
        self.target = None
        self.scalers = {}
        self.models = {}
        self.test_index = None
        
    def preparar_datos_ml(self):
        """Prepara datos específicamente para ML"""
        print("\n🔧 PREPARANDO DATOS PARA MACHINE LEARNING")
        print("="*60)
        
        # 1. Crear dataset unificado
        df_ml = self.crear_dataset_unificado()
        
        # 2. Feature engineering
        df_ml = self.crear_features_ml(df_ml)
        
        # 3. Limpieza final
        df_ml = self.limpiar_datos_ml(df_ml)
        
        # 4. Separar features y target
        self.preparar_features_target(df_ml)
        
        print(f"✅ Dataset ML preparado: {df_ml.shape}")
        print(f"📊 Features: {len(self.features)}")
        print(f"🎯 Target: {self.target}")
        
        return df_ml
    
    def crear_dataset_unificado(self):
        """Combina todas las fuentes en un dataset ML-ready"""
        print("🔄 Creando dataset unificado...")
        
        # Base: precios de maíz mensuales
        if 'agrofy_maiz' in self.eda.datasets_procesados:
            df_base = self.eda.datasets_procesados['agrofy_maiz'].resample('M')['precio_numerico'].mean()
            df_ml = pd.DataFrame({'precio_maiz': df_base})
        else:
            raise ValueError("No hay datos de maíz disponibles")
        
        # Agregar tipo de cambio
        if 'tipo_cambio' in self.eda.datasets_procesados:
            tc = self.eda.datasets_procesados['tipo_cambio']['dolar_principal'].resample('M').mean()
            df_ml = df_ml.join(tc.rename('tipo_cambio'), how='left')
        
        # Agregar IPC
        if 'ipc' in self.eda.datasets_procesados:
            ipc = self.eda.datasets_procesados['ipc']['ipc_ng_nacional'].resample('M').mean()
            df_ml = df_ml.join(ipc.rename('ipc'), how='left')
        
        # Eliminar filas con valores nulos en target
        df_ml = df_ml.dropna(subset=['precio_maiz'])
        
        print(f"📅 Período: {df_ml.index.min()} a {df_ml.index.max()}")
        print(f"📊 Registros: {len(df_ml)}")
        
        return df_ml
    
    def crear_features_ml(self, df):
        """Crea features específicas para ML"""
        print("🛠️ Creando features para ML...")
        
        df_features = df.copy()
        
        # 1. Features temporales
        df_features['año'] = df_features.index.year
        df_features['mes'] = df_features.index.month
        df_features['trimestre'] = df_features.index.quarter
        df_features['dia_año'] = df_features.index.dayofyear
        
        # Features cíclicas (mejor para ML)
        df_features['mes_sin'] = np.sin(2 * np.pi * df_features['mes'] / 12)
        df_features['mes_cos'] = np.cos(2 * np.pi * df_features['mes'] / 12)
        
        # 2. Features de lag (valores pasados)
        for lag in [1, 2, 3, 6, 12]:
            df_features[f'precio_lag_{lag}'] = df_features['precio_maiz'].shift(lag)
        
        # 3. Features estadísticas móviles
        for window in [3, 6, 12]:
            df_features[f'precio_media_{window}m'] = df_features['precio_maiz'].shift(1).rolling(window).mean()
            df_features[f'precio_std_{window}m'] = df_features['precio_maiz'].shift(1).rolling(window).std()
            df_features[f'precio_min_{window}m'] = df_features['precio_maiz'].shift(1).rolling(window).min()
            df_features[f'precio_max_{window}m'] = df_features['precio_maiz'].shift(1).rolling(window).max()
        
        # 4. Features de volatilidad
        df_features['volatilidad_3m'] = df_features['precio_maiz'].shift(1).rolling(3).std()
        df_features['volatilidad_6m'] = df_features['precio_maiz'].shift(1).rolling(6).std()
        
        # 5. Features de tendencia
        def calcular_tendencia(serie):
            if len(serie) < 2 or serie.isna().all():
                return np.nan
            x = np.arange(len(serie))
            y = serie.dropna()
            if len(y) < 2:
                return np.nan
            return np.polyfit(x[:len(y)], y, 1)[0]
        
        df_features['tendencia_3m'] = df_features['precio_maiz'].shift(1).rolling(3).apply(calcular_tendencia)
        df_features['tendencia_6m'] = df_features['precio_maiz'].shift(1).rolling(6).apply(calcular_tendencia)
        
        # 6. Features de variables externas
        if 'tipo_cambio' in df_features.columns:
            # Cambio porcentual en tipo de cambio
            df_features['tc_cambio_1m'] = df_features['tipo_cambio'].pct_change()
            df_features['tc_cambio_3m'] = df_features['tipo_cambio'].pct_change(3)
            
            # Relación precio/tipo de cambio
            df_features['precio_tc_ratio'] = df_features['precio_maiz'] / df_features['tipo_cambio']
        
        if 'ipc' in df_features.columns:
            # Inflación (cambio porcentual en IPC)
            df_features['inflacion_1m'] = df_features['ipc'].pct_change()
            df_features['inflacion_12m'] = df_features['ipc'].pct_change(12)
        
        # 7. Features de interacción
        if 'tipo_cambio' in df_features.columns and 'ipc' in df_features.columns:
            df_features['tc_ipc_ratio'] = df_features['tipo_cambio'] / df_features['ipc']
        
        print(f"✅ Features creadas: {df_features.shape[1]}")
        
        return df_features
    
    def limpiar_datos_ml(self, df):
        """Limpieza final para ML"""
        print("🧹 Limpiando datos para ML...")
        
        # Eliminar filas con demasiados valores nulos
        threshold = 0.7  # Mantener filas con al menos 70% de datos
        df_clean = df.dropna(thresh=int(threshold * len(df.columns)))
        
        # Imputar valores faltantes restantes
        # Para variables numéricas: forward fill + backward fill
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Eliminar outliers extremos (opcional)
        # df_clean = self.eliminar_outliers(df_clean)
        
        print(f"✅ Datos limpios: {df_clean.shape}")
        print(f"📊 Valores nulos restantes: {df_clean.isnull().sum().sum()}")
        
        self.df_ml = df_clean
        return df_clean
    
    def preparar_features_target(self, df):
        """Separa features y target"""
        self.target = 'precio_maiz'
        
        # Excluir target y variables que no deben ser features
        exclude_cols = [self.target, 'precio_maiz']  # Evitar data leakage
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.features = feature_cols
        
        print(f"🎯 Target: {self.target}")
        print(f"📊 Features ({len(feature_cols)}): {feature_cols[:5]}...")
    
    def crear_splits_temporales(self, test_months=6, val_months=6):
        """Crea splits respetando el orden temporal"""
        print(f"\n📊 CREANDO SPLITS TEMPORALES")
        print("="*50)
        
        df = self.df_ml.copy()
        
        # Calcular índices de corte
        total_months = len(df)
        test_start = total_months - test_months
        val_start = test_start - val_months
        
        # Crear splits
        train = df.iloc[:val_start]
        val = df.iloc[val_start:test_start]
        test = df.iloc[test_start:]
        self.test_index = test.index
        
        print(f"📈 Train: {len(train)} meses ({train.index.min()} a {train.index.max()})")
        print(f"📊 Validation: {len(val)} meses ({val.index.min()} a {val.index.max()})")
        print(f"🧪 Test: {len(test)} meses ({test.index.min()} a {test.index.max()})")
        
        return train, val, test
    
    def validacion_cruzada_temporal_xgboost(self, n_splits=5):
        print(f"\n🔁 VALIDACIÓN CRUZADA TEMPORAL CON XGBOOST")
        print("="*60)
        
        df = self.df_ml.copy()
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        mape_scores = []
        maes, rmses = [], []
        all_preds = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_idx], df.iloc[test_idx]
            X_train = train[self.features]
            y_train = train[self.target]
            X_test = test[self.features]
            y_test = test[self.target]
            
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            mape_scores.append(mape)
            rmses.append(rmse)
            maes.append(mae)
            
            print(f"Fold {fold+1}: MAPE={mape:.2f}%, RMSE={rmse:.2f}, MAE={mae:.2f}")
        
        print("\n📊 VALIDACIÓN CRUZADA FINAL:")
        print(f"• Promedio MAPE: {np.mean(mape_scores):.2f}%")
        print(f"• Promedio RMSE: {np.mean(rmses):.2f}")
        print(f"• Promedio MAE: {np.mean(maes):.2f}")

    
    def entrenar_modelos_baseline(self):
        """Entrena modelos baseline para predicción"""
        print(f"\n🤖 ENTRENANDO MODELOS BASELINE")
        print("="*50)
        
        # Preparar datos
        train, val, test = self.crear_splits_temporales()
        
        # Preparar features y target
        X_train = train[self.features]
        y_train = train[self.target]
        X_val = val[self.features]
        y_val = val[self.target]
        X_test = test[self.features]
        y_test = test[self.target]
        
        # Escalar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['features'] = scaler
        
        # Modelos a entrenar
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        
        resultados = {}
        
        for nombre, modelo in models.items():
            print(f"\n🔄 Entrenando {nombre}...")
            
            # Entrenar
            if nombre == 'Linear Regression':
                modelo.fit(X_train_scaled, y_train)
                y_pred_val = modelo.predict(X_val_scaled)
                y_pred_test = modelo.predict(X_test_scaled)
            else:
                modelo.fit(X_train, y_train)
                y_pred_val = modelo.predict(X_val)
                y_pred_test = modelo.predict(X_test)
            
            # Evaluar
            metricas_val = self.calcular_metricas(y_val, y_pred_val)
            metricas_test = self.calcular_metricas(y_test, y_pred_test)
            
            resultados[nombre] = {
                'modelo': modelo,
                'metricas_val': metricas_val,
                'metricas_test': metricas_test,
                'predicciones_val': y_pred_val,
                'predicciones_test': y_pred_test
            }
            
            print(f"✅ {nombre} - Validación MAPE: {metricas_val['MAPE']:.2f}%")
            print(f"✅ {nombre} - Test MAPE: {metricas_test['MAPE']:.2f}%")
        
        self.models = resultados
        return resultados, (X_test, y_test)
    
    def calcular_metricas(self, y_true, y_pred):
        """Calcula métricas de evaluación"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        # Precisión direccional
        if len(y_true) > 1:
            direccion_real = np.sign(np.diff(y_true))
            direccion_pred = np.sign(np.diff(y_pred))
            precision_direccion = np.mean(direccion_real == direccion_pred) * 100
        else:
            precision_direccion = np.nan
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Precision_Direccion': precision_direccion
        }
    
    def mostrar_resultados(self):
        """Muestra resultados de los modelos"""
        print(f"\n📊 RESULTADOS DE MODELOS")
        print("="*60)
        
        for nombre, resultado in self.models.items():
            print(f"\n🤖 {nombre.upper()}:")
            print("-" * 30)
            
            val_metrics = resultado['metricas_val']
            test_metrics = resultado['metricas_test']
            
            print(f"📈 VALIDACIÓN:")
            print(f"  • MAE: ${val_metrics['MAE']:.2f}")
            print(f"  • RMSE: ${val_metrics['RMSE']:.2f}")
            print(f"  • MAPE: {val_metrics['MAPE']:.2f}%")
            print(f"  • R²: {val_metrics['R2']:.3f}")
            print(f"  • Precisión Direccional: {val_metrics['Precision_Direccion']:.1f}%")
            
            print(f"🧪 TEST:")
            print(f"  • MAE: ${test_metrics['MAE']:.2f}")
            print(f"  • RMSE: ${test_metrics['RMSE']:.2f}")
            print(f"  • MAPE: {test_metrics['MAPE']:.2f}%")
            print(f"  • R²: {test_metrics['R2']:.3f}")
            print(f"  • Precisión Direccional: {test_metrics['Precision_Direccion']:.1f}%")
    
    def generar_recomendaciones_ml(self):
        """Genera recomendaciones específicas para ML evaluando múltiples métricas"""
        print("\n🎯 RECOMENDACIONES PARA PRODUCCIÓN")
        print("=" * 60)

        resultados = []

        # Recolectar métricas de cada modelo
        for nombre, info in self.models.items():
            met = info["metricas_test"]
            resultados.append({
                "nombre": nombre,
                "MAPE": met["MAPE"],
                "MAE": met["MAE"],
                "RMSE": met["RMSE"],
                "R2": met["R2"]
            })

        # Crear DataFrame para rankings
        import pandas as pd
        df = pd.DataFrame(resultados)

        # Asignar rankings (menor es mejor para errores, mayor para R2)
        df["rank_mape"] = df["MAPE"].rank(ascending=True)
        df["rank_mae"] = df["MAE"].rank(ascending=True)
        df["rank_rmse"] = df["RMSE"].rank(ascending=True)
        df["rank_r2"] = df["R2"].rank(ascending=False)

        # Puntaje final ponderado
        df["puntaje_total"] = (
            df["rank_mape"] * 0.4 +
            df["rank_mae"] * 0.2 +
            df["rank_rmse"] * 0.2 +
            df["rank_r2"] * 0.2
        )

        # Selección del mejor modelo
        mejor = df.sort_values("puntaje_total").iloc[0]

        # Mostrar resumen
        print(df[["nombre", "MAPE", "MAE", "RMSE", "R2", "puntaje_total"]])
        print(f"\n🏆 MEJOR MODELO GLOBAL: {mejor['nombre']} (Puntaje: {mejor['puntaje_total']:.2f})")

        return mejor["nombre"]


# Ejemplo de uso
def ejecutar_pipeline_ml_completo(eda_instance):
    """Ejecuta el pipeline completo de ML"""
    
    # Crear extensión ML
    ml_ext = MLExtensionMaizArcor(eda_instance)
    
    # Preparar datos
    df_ml = ml_ext.preparar_datos_ml()
    
    # Entrenar modelos
    resultados, test_data = ml_ext.entrenar_modelos_baseline()
    
    # Mostrar resultados
    ml_ext.mostrar_resultados()
    
    # Recomendaciones
    mejor_modelo = ml_ext.generar_recomendaciones_ml()
    
    return ml_ext, mejor_modelo

# Función para integrar con el código original
def extender_eda_con_ml(eda_instance):
    """Extiende el EDA original con capacidades ML"""
    print("\n🚀 EXTENDIENDO EDA CON MACHINE LEARNING")
    print("="*70)
    
    return ejecutar_pipeline_ml_completo(eda_instance)