# 🌽 Sistema Predictivo Arcor - Análisis y Predicción de Precios de Materias Primas

## 📋 Introducción

En un contexto económico marcado por la incertidumbre y la volatilidad de los precios, especialmente en materias primas, empresas como Arcor enfrentan desafíos crecientes en la gestión eficiente de sus costos. Este proyecto desarrolla un sistema predictivo basado en modelos de Inteligencia Artificial que permite anticipar variaciones en insumos clave, brindando una herramienta concreta para mejorar la toma de decisiones estratégicas.

## 🎯 Objetivos del Proyecto

- **Predicción de Precios**: Desarrollar modelos de ML para predecir precios futuros de materias primas críticas (inicialmente maíz)
- **Análisis de Volatilidad**: Identificar patrones de volatilidad y factores de riesgo en materias primas
- **Optimización de Compras**: Proporcionar insights para optimizar el timing de compras de insumos
- **Simulación de Escenarios**: Evaluar el impacto de diferentes escenarios económicos y climáticos
- **Dashboard Interactivo**: Interfaz web para visualización y análisis en tiempo real

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Frontend     │    │     Backend     │    │   Data Sources  │
│   React + Vite  │◄──►│  FastAPI + ML   │◄──►│  CSV/Excel Data │
│   TailwindCSS   │    │  Scikit-learn   │    │  Gov Datasets   │
│   Recharts      │    │  XGBoost        │    │  Market Data    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Fuentes de Datos

### Datasets Utilizados

1. **Datos de Precios de Maíz (Agrofy)**
   - Archivo: `series-historicas-pizarra.csv`
   - Fuente: https://news.agrofy.com.ar/granos/series-historicas-pizarra
   - Descripción: Precios históricos de maíz en mercados argentinos

2. **Tipos de Cambio Históricos (BCRA)**
   - Archivo: `tipos-de-cambio-historicos.csv`
   - Fuente: https://datos.gob.ar/dataset/sspm-tipos-cambio-historicos
   - Descripción: Cotizaciones históricas del dólar estadounidense

3. **Índice de Precios al Consumidor (INDEC)**
   - Archivo: `indice-precios-al-consumidor-nivel-general-base-diciembre-2016-mensual.csv`
   - Fuente: https://datos.gob.ar/dataset/sspm-indice-precios-al-consumidor-nacional-ipc
   - Descripción: IPC nacional base diciembre 2016

4. **Datos Climáticos (SMN)**
   - Archivo: `Estadísticas normales Datos abiertos 1991-2020.xlsx`
   - Fuente: https://www.smn.gob.ar/descarga-de-datos
   - Descripción: Estadísticas meteorológicas normales 1991-2020

5. **Datos Adicionales de Maíz (MAGYP)**
   - Archivo: `maiz_ecc.csv`
   - Fuente: https://datos.magyp.gob.ar/dataset/maiz-siembra-cosecha-rendimiento-produccion
   - Descripción: Datos de siembra, cosecha y producción de maíz

## 🛠️ Tecnologías y Librerías

### Backend (Python)

#### Librerías de Machine Learning y Análisis de Datos
```python
# Machine Learning
scikit-learn==1.7.0          # Modelos ML principales
xgboost==3.0.2               # Gradient Boosting avanzado
numpy==2.3.0                 # Operaciones numéricas
pandas==2.3.0                # Manipulación de datos

# Visualización y Análisis
matplotlib==3.10.3           # Gráficos básicos
seaborn==0.13.2              # Visualizaciones estadísticas
scipy==1.15.3                # Funciones científicas

# API y Web
fastapi==0.115.12            # Framework web moderno
uvicorn==0.34.3              # Servidor ASGI
pydantic==2.11.7             # Validación de datos

# Utilidades
openpyxl==3.1.5              # Lectura de archivos Excel
python-dateutil==2.9.0       # Manipulación de fechas
pytz==2025.2                 # Zonas horarias
```

#### Modelos de Machine Learning Implementados
- **Random Forest Regressor**: Ensemble de árboles para capturar relaciones no lineales
- **XGBoost**: Gradient boosting optimizado para predicciones robustas
- **Ridge Regression**: Regresión lineal con regularización L2
- **Lasso Regression**: Regresión con selección automática de features
- **Elastic Net**: Combinación de Ridge y Lasso
- **Gradient Boosting**: Ensemble secuencial para reducir sesgo

### Frontend (React)

#### Librerías Principales
```json
{
  "react": "^19.1.0",              // Framework de interfaz de usuario
  "react-dom": "^19.1.0",          // Manipulación del DOM
  "recharts": "^2.15.3",           // Visualizaciones interactivas
  "tailwindcss": "^4.1.10",        // Framework CSS utility-first
  "lucide-react": "^0.515.0",      // Íconos modernos
  "vite": "^6.3.5"                 // Bundler y servidor de desarrollo
}
```

## 📁 Estructura del Proyecto

```
TP_Ciencia_Datos/
├── 📂 backend/                    # Servidor API y lógica ML
│   ├── 📂 datos/                  # Datasets y fuentes de datos
│   │   ├── series-historicas-pizarra.csv
│   │   ├── tipos-de-cambio-historicos.csv
│   │   ├── maiz_ecc.csv
│   │   ├── indice-precios-al-consumidor-nivel-general-base-diciembre-2016-mensual.csv
│   │   └── Estadísticas normales Datos abiertos 1991-2020.xlsx
│   ├── 🐍 main.py                 # Punto de entrada principal
│   ├── 🐍 api.py                  # Endpoints de la API REST
│   ├── 🐍 eda.py                  # Análisis Exploratorio de Datos
│   ├── 🐍 MLExtension.py          # Pipeline de Machine Learning
│   ├── 🐍 visualizaciones_arcor.py # Gráficos y visualizaciones
│   ├── 📄 requirements.txt        # Dependencias Python
│   ├── 📄 datasets.txt           # URLs de fuentes de datos
│   └── 📄 README.md              # Documentación del backend
├── 📂 frontend/                   # Interfaz web React
│   ├── 📂 src/                    # Código fuente React
│   │   ├── ⚛️ main.jsx            # Punto de entrada React
│   │   ├── ⚛️ App.jsx             # Componente principal
│   │   ├── ⚛️ ArcorPredictiveSystem.jsx # Dashboard principal
│   │   ├── 🎨 index.css           # Estilos globales
│   │   └── 🎨 App.css             # Estilos específicos
│   ├── 📂 public/                 # Archivos estáticos
│   ├── 📄 package.json           # Dependencias Node.js
│   ├── 📄 vite.config.js         # Configuración Vite
│   ├── 📄 eslint.config.js       # Configuración ESLint
│   └── 📄 index.html             # Template HTML
├── 📂 .devcontainer/             # Configuración de desarrollo
├── 🐳 compose.yml                # Orquestación Docker
└── 📄 README.md                  # Este archivo
```

## 🔍 Descripción Detallada de Archivos

### Backend

#### `main.py` - Orquestador Principal
- **Función**: Punto de entrada que ejecuta el pipeline completo
- **Responsabilidades**:
  - Inicialización de datasets
  - Ejecución del EDA
  - Entrenamiento de modelos ML
  - Generación de visualizaciones

#### `eda.py` - Análisis Exploratorio de Datos (EDA)
- **Clase Principal**: `EDAMaizArcorMejorado`
- **Funcionalidades**:
  - Carga y limpieza de datasets heterogéneos
  - Análisis estadístico descriptivo
  - Detección de patrones estacionales
  - Análisis de correlaciones entre variables
  - Generación de insights predictivos
- **Métricas Calculadas**:
  - Volatilidad histórica
  - Correlaciones cruzadas
  - Análisis de tendencias
  - Detección de outliers

#### `MLExtension.py` - Pipeline de Machine Learning
- **Clase Principal**: `EnhancedRobustCornPipeline`
- **Características Avanzadas**:
  - **Feature Engineering Inteligente**:
    - Variables temporales (estacionalidad, tendencias)
    - Lags de precios (1, 2, 3, 6, 12 meses)
    - Medias móviles (3, 6, 12 meses)
    - Ratios de volatilidad
    - Indicadores de mercado (bull/bear)
  - **Preprocesamiento Robusto**:
    - Manejo de valores infinitos y extremos
    - Imputación inteligente (KNN, mediana)
    - Escalado robusto (RobustScaler)
    - Selección automática de features
  - **Validación Temporal**:
    - TimeSeriesSplit para respeta la secuencia temporal
    - Validación cruzada específica para series temporales
  - **Múltiples Modelos**:
    - Entrenamiento paralelo de 6 algoritmos diferentes
    - Selección automática del mejor modelo
    - Métricas de evaluación: MAPE, R², RMSE, MAE

#### `api.py` - API REST con FastAPI
- **Endpoints Principales**:
  - `GET /api/predicciones/maiz`: Predicciones históricas vs reales
  - `GET /api/prediccion`: Predicciones futuras con intervalos de confianza
  - `GET /api/insights`: Insights del análisis exploratorio
  - `GET /api/escenarios`: Simulación de escenarios económicos
  - `GET /api/evaluacion_modelos`: Métricas de performance de modelos
  - `GET /api/feature_importance`: Importancia de variables
  - `GET /api/diagnosticos`: Diagnóstico de calidad del modelo
- **Características**:
  - CORS configurado para desarrollo
  - Manejo robusto de errores
  - Conversión automática de tipos numpy
  - Simulación de escenarios hipotéticos

#### `visualizaciones_arcor.py` - Generación de Gráficos
- **Funciones**:
  - Comparación real vs predicho por modelo
  - Análisis de errores absolutos
  - MAPE global por modelo
  - Importancia de variables (para modelos tree-based)

### Frontend

#### `ArcorPredictiveSystem.jsx` - Dashboard Principal
- **Componentes**:
  - **MetricCard**: Tarjetas de métricas con tendencias
  - **AlertCard**: Sistema de alertas y notificaciones
  - **Gráficos Interactivos**: 
    - Líneas temporales de precios y predicciones
    - Intervalos de confianza
    - Comparación de escenarios
- **Estados Manejados**:
  - Datos históricos y predicciones
  - Evaluación de modelos
  - Escenarios simulados
  - Alertas en tiempo real
  - Estados de carga y error

#### Características del Dashboard
- **Responsive Design**: Adaptable a diferentes tamaños de pantalla
- **Interactividad**: Gráficos interactivos con Recharts
- **Sistema de Alertas**: Notificaciones automáticas de cambios significativos
- **Métricas en Tiempo Real**: Visualización de KPIs principales
- **Simulación de Escenarios**: Análisis de impacto de diferentes variables

## 🔬 Metodología de Ciencia de Datos

### 1. Adquisición y Preparación de Datos
- **Integración Multi-fuente**: Combinación de datasets gubernamentales y privados
- **Limpieza Inteligente**: Detección y corrección de outliers, valores faltantes
- **Sincronización Temporal**: Alineación de series temporales con diferentes frecuencias

### 2. Feature Engineering Avanzado
- **Variables Temporales**: Estacionalidad, tendencias, ciclos
- **Variables Lag**: Precios retrasados para capturar memoria del mercado
- **Indicadores Técnicos**: Medias móviles, volatilidad, momentum
- **Variables Externas**: Tipo de cambio, IPC, datos climáticos
- **Variables de Régimen**: Identificación de mercados alcistas/bajistas

### 3. Modelado Predictivo
- **Ensemble Learning**: Combinación de múltiples algoritmos
- **Validación Temporal**: Respeto de la estructura temporal de los datos
- **Optimización de Hiperparámetros**: Grid search con validación cruzada
- **Selección de Modelos**: Basada en múltiples métricas (MAPE, R², estabilidad)

### 4. Evaluación y Diagnóstico
- **Métricas Múltiples**:
  - MAPE (Mean Absolute Percentage Error)
  - R² (Coeficiente de determinación)
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
- **Análisis de Residuos**: Verificación de supuestos del modelo
- **Pruebas de Estabilidad**: Consistencia en diferentes períodos
- **Intervalos de Confianza**: Cuantificación de incertidumbre

## 🚀 Instalación y Ejecución

### Requisitos Previos
- Docker y Docker Compose
- Python 3.13+ (si se ejecuta localmente)
- Node.js 24+ (si se ejecuta localmente)

### Ejecución con Docker (Recomendado)
```bash
# Clonar el repositorio
git clone <repository-url>
cd TP_Ciencia_Datos

# Ejecutar con Docker Compose
docker-compose up --build

# Acceder a la aplicación
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# Documentación API: http://localhost:8000/docs
```

### Ejecución Local

#### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 📈 Resultados y Métricas

### Performance de Modelos
- **Mejor Modelo**: XGBoost (típicamente)
- **MAPE Promedio**: < 15% (objetivo: predicciones con error menor al 15%)
- **R² Score**: > 0.7 (explicación del 70%+ de la varianza)
- **Horizonte de Predicción**: 1-12 meses
- **Intervalos de Confianza**: 95% de confianza estadística

### Insights Clave Identificados
- **Estacionalidad**: Patrones claros en precios según épocas de siembra/cosecha
- **Correlación Cambiaria**: Fuerte correlación con el tipo de cambio USD/ARS
- **Impacto Climático**: Variables meteorológicas significativas en la predicción
- **Volatilidad**: Períodos de alta volatilidad asociados a eventos macroeconómicos

## 🔮 Simulación de Escenarios

El sistema permite simular diferentes escenarios económicos:

### Escenarios Implementados
1. **Devaluación**: Impacto de cambios en el tipo de cambio
2. **Sequía**: Efecto de condiciones climáticas adversas
3. **Inflación**: Variaciones en el IPC y su impacto en precios
4. **Estacionalidad**: Efectos de diferentes épocas del año

### Métricas de Escenarios
- **Precio Base**: Predicción bajo condiciones normales
- **Variación Porcentual**: Cambio esperado por escenario
- **Intervalo de Impacto**: Rango de variación posible
- **Probabilidad**: Estimación de probabilidad del escenario

## 🎯 Casos de Uso para Arcor

### 1. Optimización de Compras
- **Timing de Compras**: Identificar momentos óptimos para comprar materias primas
- **Volúmenes**: Ajustar volúmenes según predicciones de precio
- **Contratos**: Evaluar conveniencia de contratos a futuro vs spot

### 2. Gestión de Riesgo
- **Cobertura Financiera**: Determinar necesidad de instrumentos de cobertura
- **Diversificación**: Evaluar proveedores alternativos
- **Contingencias**: Preparar planes para escenarios adversos

### 3. Planificación Estratégica
- **Presupuestos**: Mejorar precisión en proyecciones de costos
- **Pricing**: Ajustar precios de productos finales anticipadamente
- **Inversiones**: Evaluar conveniencia de integración vertical

## 🔄 Extensibilidad y Futuro

### Materias Primas Adicionales
El sistema está diseñado para ser fácilmente extensible a otras materias primas:
- **Azúcar**: Segundo insumo más importante para Arcor
- **Leche en Polvo**: Crítico para productos lácteos
- **Trigo**: Para productos de panificación

### Mejoras Futuras
- **Datos en Tiempo Real**: Integración con APIs de mercados financieros
- **Deep Learning**: Implementación de redes neuronales para patrones complejos
- **Análisis de Sentimiento**: Incorporación de noticias y redes sociales
- **Optimización Multi-objetivo**: Balancear múltiples criterios simultáneamente

## 📊 Monitoreo y Mantenimiento

### KPIs del Sistema
- **Precisión de Predicciones**: Seguimiento continuo del MAPE
- **Drift de Datos**: Detección de cambios en distribuciones
- **Performance de API**: Tiempos de respuesta y disponibilidad
- **Actualización de Datos**: Frecuencia y calidad de nuevos datos

### Alertas Automáticas
- **Desviaciones Significativas**: Cuando predicciones vs realidad > umbral
- **Cambios de Tendencia**: Detección de cambios de régimen de mercado
- **Problemas de Datos**: Faltantes o inconsistencias en datos de entrada
- **Performance del Modelo**: Degradación en métricas de evaluación

## 👥 Equipo y Contribuciones

Este proyecto representa un trabajo integral de ciencia de datos aplicada a la industria alimentaria, combinando:
- **Análisis Exploratorio**: Comprensión profunda de los datos y el dominio
- **Machine Learning**: Implementación de algoritmos estado del arte
- **Ingeniería de Software**: API robusta y escalable
- **Visualización de Datos**: Dashboard interactivo para usuarios finales
- **Conocimiento del Negocio**: Aplicación práctica a problemas reales de Arcor

## 📚 Referencias y Fuentes

- Banco Central de la República Argentina (BCRA)
- Instituto Nacional de Estadística y Censos (INDEC)
- Ministerio de Agricultura, Ganadería y Pesca (MAGYP)
- Servicio Meteorológico Nacional (SMN)
- Agrofy - Plataforma de Agro Tecnología
- Scikit-learn Documentation
- XGBoost Documentation
- FastAPI Documentation
- React Documentation

---

**Desarrollado como parte del Trabajo Práctico de Ciencia de Datos**  
*Sistema Predictivo para Optimización de Compras de Materias Primas - Grupo Arcor*
