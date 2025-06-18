
# 🌽 Sistema Predictivo de Precios de Maíz - Grupo Arcor

Este proyecto implementa un sistema completo de análisis exploratorio y modelado predictivo para anticipar los precios del maíz en Argentina. Integra procesamiento de datos, análisis visual, ingeniería de features y entrenamiento de modelos de machine learning. Se orienta a brindar soporte en la toma de decisiones de compra para Grupo Arcor.

---

## 📦 Estructura del Proyecto

```
├── main.py               # Script principal para ejecutar el pipeline completo
├── api.py                # API REST construida con FastAPI
├── eda.py                # Módulo de análisis exploratorio de datos (EDA)
├── MLExtension.py        # Clase principal del pipeline de ML mejorado
├── datasets.txt          # Fuentes de los datasets utilizados
├── datos/                # Carpeta sugerida para almacenar los CSV originales
│   ├── tipos-de-cambio-historicos.csv
│   ├── maiz_ecc.csv
│   ├── indice-precios-al-consumidor-nivel-general-base-diciembre-2016-mensual.csv
│   ├── Estadísticas normales Datos abiertos 1991-2020.xlsx
│   └── series-historicas-pizarra.csv
```

---

## 🚀 Funcionalidades

- **EDA mejorado** con visualizaciones y análisis descriptivos.
- **Ingeniería de features** avanzada (lags, estacionalidad, volatilidad, etc.).
- **Modelos ML robustos**: Random Forest, HistGradientBoosting, Ridge, Lasso, ElasticNet, y XGBoost (si está disponible).
- **Evaluación automática** y selección del mejor modelo.
- **Predicción futura** con incertidumbre y escenarios simulados.
- **API REST** para consumo desde un frontend.

---

## 📊 Datasets Utilizados

| Fuente | Descripción |
|-------|-------------|
| [Precios Maíz Agrofy](https://news.agrofy.com.ar/granos/series-historicas-pizarra) | Precios históricos del maíz |
| [Tipo de Cambio Histórico](https://datos.gob.ar/dataset/sspm-tipos-cambio-historicos/archivo/sspm_175.1) | Tipo de cambio oficial y dólar |
| [IPC Nacional](https://datos.gob.ar/dataset/sspm-indice-precios-al-consumidor-nacional-ipc-base-diciembre-2016/archivo/sspm_145.3) | Inflación mensual Argentina |
| [Datos Climáticos](https://www.smn.gob.ar/descarga-de-datos) | Estadísticas climáticas históricas |
| [Producción de Maíz MAGyP](https://datos.magyp.gob.ar/dataset/maiz-siembra-cosecha-rendimiento-produccion/archivo/c4696e28-4e54-464b-bc3b-19c608371231) | Información agrícola complementaria |

---

## 🧠 Modelado Predictivo

- **Target**: Precio mensual promedio del maíz.
- **Features generadas**: estacionales, tendencia, lags, rolling stats, tipo de cambio, volatilidad, ciclos agrícolas, etc.
- **Selección inteligente de features**: eliminación de baja varianza y alta colinealidad.
- **Entrenamiento con validación temporal**.
- **Métricas reportadas**: MAE, RMSE, MAPE, R².

---

## 🔌 API Endpoints

| Endpoint | Descripción |
|---------|-------------|
| `/api/insights` | Devuelve insights del EDA y ML |
| `/api/predicciones/maiz` | Predicciones vs reales (últimos 24 meses) |
| `/api/prediccion?horizonte=3` | Proyección futura (default: 3 meses) |
| `/api/escenarios` | Predicciones con escenarios simulados |
| `/api/modelo` | Métricas del mejor modelo |
| `/api/evaluacion_modelos` | Comparación de todos los modelos entrenados |
| `/api/feature_importance` | Importancia de features |
| `/api/historico/maiz` | Datos históricos de precios |
| `/api/diagnosticos` | Diagnóstico de calidad de datos y modelo |
| `/api/health` | Endpoint de salud de la API |

---

## ⚙️ Instalación y Uso

1. **Clonar repositorio**  
   ```bash
   git clone <url-del-repo>
   cd sistema-predictivo-maiz
   ```

2. **Instalar dependencias**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Colocar datasets en carpeta `datos/`** según lo indicado en `datasets.txt`.

4. **Ejecutar pipeline**  
   ```bash
   python main.py
   ```

5. **Iniciar API**  
   ```bash
   uvicorn api:app --reload
   ```

6. **Probar en navegador o Postman**  
   Visitar: `http://localhost:8000/docs`

---

## 📈 Visualizaciones

El módulo `eda.py` genera automáticamente gráficos de:

- Evolución y distribución de precios
- Estacionalidad
- Correlaciones con tipo de cambio e IPC
- Volatilidad histórica

---

## 👥 Autores

- Equipo de Ciencia de Datos – Grupo Arcor  
- Colaboración: Universidad / Consultora externa

---

## 📄 Licencia

Uso interno para fines analíticos y predictivos. No distribuir sin autorización de Grupo Arcor.
