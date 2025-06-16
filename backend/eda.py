import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import re
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

class EDAMaizArcorMejorado:
    def __init__(self, dataset_paths):
        self.dataset_paths = dataset_paths
        self.datasets = {}
        self.datasets_procesados = {}
        
    def cargar_datos(self):
        """Carga todos los datasets con manejo mejorado de errores"""
        print("🔄 Cargando datasets...")
        
        # Tipo de cambio
        try:
            self.datasets['tipo_cambio'] = pd.read_csv(self.dataset_paths['tipo_cambio'])
            print("✅ Tipo de cambio cargado")
        except Exception as e:
            print(f"❌ Error cargando tipo_cambio: {e}")
            
        # Datos de maíz - probar diferentes encodings
        try:
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    self.datasets['maiz_datos'] = pd.read_csv(self.dataset_paths['maiz_datos'], encoding=encoding)
                    print(f"✅ Datos de maíz cargados con encoding: {encoding}")
                    break
                except:
                    continue
            else:
                print("❌ No se pudo cargar datos de maíz con ningún encoding")
        except Exception as e:
            print(f"❌ Error cargando maiz_datos: {e}")
            
        # IPC
        try:
            self.datasets['ipc'] = pd.read_csv(self.dataset_paths['ipc'])
            print("✅ IPC cargado")
        except Exception as e:
            print(f"❌ Error cargando IPC: {e}")
            
        # Clima - mejorar lectura de Excel
        try:
            # Leer múltiples hojas si existen
            excel_file = pd.ExcelFile(self.dataset_paths['clima'])
            print(f"Hojas disponibles en clima: {excel_file.sheet_names}")
            # Leer la primera hoja con datos
            self.datasets['clima'] = pd.read_excel(self.dataset_paths['clima'], sheet_name=0, header=3)
            print("✅ Datos climáticos cargados")
        except Exception as e:
            print(f"❌ Error cargando clima: {e}")
            
        # Precios Agrofy - parsing correcto del CSV
        try:
            # El archivo parece tener formato especial con separadores ;
            self.datasets['agrofy_precios'] = pd.read_csv(self.dataset_paths['agrofy_precios'], 
                                                         sep=';', 
                                                         names=['Fecha', 'Mercado', 'Producto', 'Precio'],
                                                         skiprows=0)
            # Si no funciona, probar como una sola columna y parsear manualmente
            if len(self.datasets['agrofy_precios'].columns) == 1:
                df_temp = pd.read_csv(self.dataset_paths['agrofy_precios'], header=None)
                # Parsear la columna única
                parsed_data = []
                for row in df_temp.iloc[:, 0]:
                    if pd.notna(row) and ';' in str(row):
                        parts = str(row).split(';')
                        if len(parts) >= 4:
                            parsed_data.append({
                                'Fecha': parts[0].strip(),
                                'Mercado': parts[1].strip().replace('"', ''),
                                'Producto': parts[2].strip().replace('"', ''),
                                'Precio': parts[3].strip().replace('"', '')
                            })
                self.datasets['agrofy_precios'] = pd.DataFrame(parsed_data)
            print("✅ Precios Agrofy cargados y parseados")
        except Exception as e:
            print(f"❌ Error cargando agrofy_precios: {e}")
    
    def procesar_datasets(self):
        """Procesa y limpia los datasets cargados"""
        print("\n🔧 PROCESANDO DATASETS...")
        print("="*50)
        
        # Procesar tipo de cambio
        if 'tipo_cambio' in self.datasets:
            df = self.datasets['tipo_cambio'].copy()
            # Convertir fecha
            df['fecha'] = pd.to_datetime(df['indice_tiempo'])
            df = df.set_index('fecha')
            # Usar el dólar estadounidense como principal (mejor cobertura)
            df['dolar_principal'] = df['dolar_estadounidense'].fillna(df['dolar_oficial_venta'])
            self.datasets_procesados['tipo_cambio'] = df
            print("✅ Tipo de cambio procesado")
        
        # Procesar IPC
        if 'ipc' in self.datasets:
            df = self.datasets['ipc'].copy()
            df['fecha'] = pd.to_datetime(df['indice_tiempo'])
            df = df.set_index('fecha')
            self.datasets_procesados['ipc'] = df
            print("✅ IPC procesado")
        
        # Procesar Agrofy (MAÍZ)
        if 'agrofy_precios' in self.datasets:
            df = self.datasets['agrofy_precios'].copy()
            # Convertir fecha (formato dd-mm-yy)
            try:
                df['fecha'] = pd.to_datetime(df['Fecha'], format='%d-%m-%y', errors='coerce')
                # Si hay fechas con formato de 4 dígitos en año
                mask_null = df['fecha'].isna()
                if mask_null.sum() > 0:
                    df.loc[mask_null, 'fecha'] = pd.to_datetime(df.loc[mask_null, 'Fecha'], 
                                                               format='%d-%m-%Y', errors='coerce')
            except:
                print("⚠️ Problema con formato de fechas en Agrofy")
            
            # Extraer precio numérico
            df['precio_numerico'] = df['Precio'].str.extract(r'([0-9,\.]+)').astype(str)
            df['precio_numerico'] = df['precio_numerico'].str.replace(',', '.').astype(float)
            
            # Filtrar solo maíz
            df_maiz = df[df['Producto'].str.contains('Maiz|maiz|MAIZ', case=False, na=False)]
            df_maiz = df_maiz.set_index('fecha').sort_index()
            
            self.datasets_procesados['agrofy_maiz'] = df_maiz
            print(f"✅ Agrofy procesado - {len(df_maiz)} registros de maíz")
        
        # Procesar clima
        if 'clima' in self.datasets:
            df = self.datasets['clima'].copy()
            # Limpiar y procesar datos climáticos
            # Buscar filas con datos de estaciones
            df_clean = df.dropna(how='all')
            self.datasets_procesados['clima'] = df_clean
            print("✅ Clima procesado")
        
        # Procesar maiz_datos si se cargó
        if 'maiz_datos' in self.datasets:
            df = self.datasets['maiz_datos'].copy()
            self.datasets_procesados['maiz_datos'] = df
            print("✅ Maíz datos procesado")

    def analisis_datasets(self):
        print("\n" + "="*60)
        print("📊 ANÁLISIS DESCRIPTIVO DE TODOS LOS DATASETS")
        print("="*60)
        
        if not self.datasets_procesados:
            print("❌ No hay datasets procesados para analizar")
            return
        
        for nombre, df in self.datasets_procesados.items():
            print(f"\n🔹 Dataset: {nombre}")
            print("-" * 40)
            
            # Información general
            print(f"• Período de datos: {df.index.min()} a {df.index.max()}" if hasattr(df.index, 'min') else "• Índice no es fecha")
            print(f"• Total de registros: {len(df):,}")
            print(f"• Columnas: {list(df.columns)}")
            
            # Estadísticas numéricas
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                print("\n  → Estadísticas numéricas:")
                desc_num = df[num_cols].describe().T
                for col in desc_num.index:
                    stats = desc_num.loc[col]
                    print(f"    • {col}: mean={stats['mean']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}, std={stats['std']:.2f}")
            else:
                print("  → No hay columnas numéricas")
            
            # Estadísticas categóricas
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                print("\n  → Estadísticas categóricas (top 3 valores más frecuentes):")
                for col in cat_cols:
                    top_vals = df[col].value_counts().head(3)
                    print(f"    • {col}:")
                    for val, cnt in top_vals.items():
                        print(f"      - {val}: {cnt} registros")
            else:
                print("  → No hay columnas categóricas")
    
    def analisis_maiz_detallado(self):
        """Análisis específico y detallado del maíz"""
        print("\n" + "="*60)
        print("🌽 ANÁLISIS DETALLADO DEL MAÍZ")
        print("="*60)
        
        if 'agrofy_maiz' in self.datasets_procesados:
            df_maiz = self.datasets_procesados['agrofy_maiz']
            
            print(f"\n📊 ESTADÍSTICAS DEL MAÍZ:")
            print("-" * 40)
            print(f"• Período de datos: {df_maiz.index.min()} a {df_maiz.index.max()}")
            print(f"• Total de registros: {len(df_maiz):,}")
            print(f"• Mercados: {df_maiz['Mercado'].unique()}")
            
            print(f"\n💰 PRECIOS:")
            print("-" * 20)
            precios_stats = df_maiz['precio_numerico'].describe()
            print(f"• Precio promedio: U$S {precios_stats['mean']:.2f}")
            print(f"• Precio mínimo: U$S {precios_stats['min']:.2f}")
            print(f"• Precio máximo: U$S {precios_stats['max']:.2f}")
            print(f"• Desviación estándar: U$S {precios_stats['std']:.2f}")
            
            # Análisis por mercado
            print(f"\n🏢 ANÁLISIS POR MERCADO:")
            print("-" * 30)
            mercado_stats = df_maiz.groupby('Mercado')['precio_numerico'].agg(['count', 'mean', 'std'])
            print(mercado_stats.round(2))
            
            # Tendencia temporal
            print(f"\n📈 TENDENCIA TEMPORAL:")
            print("-" * 25)
            df_maiz_monthly = df_maiz.resample('M')['precio_numerico'].mean()
            print(f"• Primer año promedio: U$S {df_maiz_monthly.iloc[:12].mean():.2f}")
            print(f"• Último año promedio: U$S {df_maiz_monthly.iloc[-12:].mean():.2f}")
            
            variacion = ((df_maiz_monthly.iloc[-12:].mean() / df_maiz_monthly.iloc[:12].mean()) - 1) * 100
            print(f"• Variación total: {variacion:.1f}%")
            
            return df_maiz
        else:
            print("❌ No hay datos de maíz procesados para analizar")
            return None
    
    def crear_visualizaciones_mejoradas(self):
        """Crea visualizaciones mejoradas y específicas"""
        print("\n" + "="*60)
        print("📊 CREANDO VISUALIZACIONES MEJORADAS")
        print("="*60)
        
        # Crear figura con subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle('🌽 ANÁLISIS EXPLORATORIO - PREDICCIÓN PRECIOS MAÍZ ARCOR', fontsize=16, fontweight='bold')
        
        # 1. Evolución precio del maíz
        if 'agrofy_maiz' in self.datasets_procesados:
            df_maiz = self.datasets_procesados['agrofy_maiz']
            df_maiz_monthly = df_maiz.resample('M')['precio_numerico'].mean()
            
            axes[0,0].plot(df_maiz_monthly.index, df_maiz_monthly.values, linewidth=2, color='green')
            axes[0,0].set_title('📈 Evolución Precio Maíz (Mensual)', fontweight='bold')
            axes[0,0].set_ylabel('Precio (U$S)')
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Distribución de precios del maíz
        if 'agrofy_maiz' in self.datasets_procesados:
            df_maiz = self.datasets_procesados['agrofy_maiz']
            precio_actual = df_maiz['precio_numerico'].iloc[-1]  # Último precio disponible

            axes[0,1].hist(df_maiz['precio_numerico'], bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[0,1].axvline(precio_actual, color='red', linestyle='--', linewidth=2, label=f'Precio actual: U$S {precio_actual:.2f}')
            axes[0,1].legend()
            axes[0,1].set_title('📊 Distribución Precios Maíz (Histograma)', fontweight='bold')
            axes[0,1].set_xlabel('Precio (U$S)')
            axes[0,1].set_ylabel('Frecuencia')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Precios por mercado
        if 'agrofy_maiz' in self.datasets_procesados:
            df_maiz = self.datasets_procesados['agrofy_maiz']
            
            # Filtro para eliminar outliers extremos
            df_maiz_filtrado = df_maiz[df_maiz['precio_numerico'] <= 500]

            df_maiz_filtrado.boxplot(
                column='precio_numerico',
                by='Mercado',
                ax=axes[0,2],
                patch_artist=True,  # permite aplicar color
                boxprops=dict(facecolor='lightgreen', color='black'),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(markerfacecolor='red', marker='o', markersize=3, linestyle='none')
            )
            axes[0,2].set_title('Precios por Mercado (sin outliers extremos)', fontweight='bold')
            axes[0,2].set_xlabel('Mercado')
            axes[0,2].set_ylabel('Precio (U$S)')
            axes[0,2].tick_params(axis='x', rotation=45)

        
        # 4. Evolución tipo de cambio
        if 'tipo_cambio' in self.datasets_procesados:
            df_tc = self.datasets_procesados['tipo_cambio']
            df_tc_monthly = df_tc['dolar_principal'].resample('M').mean()
            # Filtrar últimos 5 años para mejor visualización
            df_tc_recent = df_tc_monthly.last('60M')
            
            axes[1,0].plot(df_tc_recent.index, df_tc_recent.values, linewidth=2, color='blue')
            axes[1,0].set_title('💱 Evolución Tipo de Cambio (5 años)', fontweight='bold')
            axes[1,0].set_ylabel('Pesos por Dólar')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. Evolución IPC
        if 'ipc' in self.datasets_procesados:
            df_ipc = self.datasets_procesados['ipc']
            axes[1,1].plot(df_ipc.index, df_ipc['ipc_ng_nacional_tasa_variacion_mensual'], linewidth=2, color='red')
            axes[1,1].set_title('📉 Inflación Mensual (IPC Nacional)', fontweight='bold')
            axes[1,1].set_ylabel('Variación Mensual (%)')
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].tick_params(axis='x', rotation=45)

        
        # 6. Correlación precio maíz vs tipo de cambio
        if 'agrofy_maiz' in self.datasets_procesados and 'tipo_cambio' in self.datasets_procesados:
            df_maiz = self.datasets_procesados['agrofy_maiz'].resample('M')['precio_numerico'].mean()
            df_tc = self.datasets_procesados['tipo_cambio']['dolar_principal'].resample('M').mean()
            
            df_combined = pd.DataFrame({'maiz': df_maiz, 'tipo_cambio': df_tc}).dropna()
            
            if len(df_combined) > 0:
                axes[1,2].scatter(df_combined['tipo_cambio'], df_combined['maiz'], alpha=0.6, color='purple', label='Observaciones')
                
                # Línea de regresión
                m, b = np.polyfit(df_combined['tipo_cambio'], df_combined['maiz'], 1)
                axes[1,2].plot(df_combined['tipo_cambio'], m*df_combined['tipo_cambio'] + b, color='orange', linestyle='--', label='Regresión lineal')
                axes[1,2].legend()
                
                axes[1,2].set_title('🔗 Maíz vs Tipo de Cambio', fontweight='bold')
                axes[1,2].set_xlabel('Tipo de Cambio')
                axes[1,2].set_ylabel('Precio Maíz (U$S)')
                axes[1,2].grid(True, alpha=0.3)
                
                corr = df_combined['maiz'].corr(df_combined['tipo_cambio'])
                axes[1,2].text(0.05, 0.95, f'Correlación: {corr:.3f}', 
                            transform=axes[1,2].transAxes, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

        
        # 7. Estacionalidad del maíz
        if 'agrofy_maiz' in self.datasets_procesados:
            df_maiz = self.datasets_procesados['agrofy_maiz']
            df_maiz['mes'] = df_maiz.index.month
            monthly_avg = df_maiz.groupby('mes')['precio_numerico'].mean()
            
            axes[2,0].bar(monthly_avg.index, monthly_avg.values, color='green', alpha=0.7)
            axes[2,0].set_title('🗓️ Estacionalidad Precios Maíz', fontweight='bold')
            axes[2,0].set_xlabel('Mes')
            axes[2,0].set_ylabel('Precio Promedio (U$S)')
            axes[2,0].set_xticks(range(1, 13))
            axes[2,0].set_xticklabels(['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'])
            axes[2,0].grid(True, alpha=0.3)

        
        # 8. Volatilidad del precio del maíz
        if 'agrofy_maiz' in self.datasets_procesados:
            df_maiz = self.datasets_procesados['agrofy_maiz']
            df_maiz_monthly = df_maiz.resample('M')['precio_numerico'].agg(['mean', 'std'])
            df_maiz_monthly['std_smooth'] = df_maiz_monthly['std'].rolling(window=3).mean()
            
            axes[2,1].plot(df_maiz_monthly.index, df_maiz_monthly['std'], linewidth=1, color='red', alpha=0.5, label='Volatilidad')
            axes[2,1].plot(df_maiz_monthly.index, df_maiz_monthly['std_smooth'], linewidth=2, color='darkred', label='Suavizada (3 meses)')
            
            axes[2,1].set_title('📊 Volatilidad Precio Maíz', fontweight='bold')
            axes[2,1].set_ylabel('Desviación Estándar')
            axes[2,1].grid(True, alpha=0.3)
            axes[2,1].tick_params(axis='x', rotation=45)
            axes[2,1].legend()
        
        # 9. Resumen estadístico
        axes[2,2].axis('off')
        
        # Texto de resumen
        resumen_text = "📋 RESUMEN EJECUTIVO\n\n"
        
        if 'agrofy_maiz' in self.datasets_procesados:
            df_maiz = self.datasets_procesados['agrofy_maiz']
            precio_actual = df_maiz['precio_numerico'].iloc[-1]
            precio_min = df_maiz['precio_numerico'].min()
            precio_max = df_maiz['precio_numerico'].max()
            
            resumen_text += f"🌽 MAÍZ:\n"
            resumen_text += f"• Precio actual: U$S {precio_actual:.2f}\n"
            resumen_text += f"• Rango histórico: U$S {precio_min:.2f} - {precio_max:.2f}\n"
            resumen_text += f"• Registros: {len(df_maiz):,}\n\n"
        
        if 'tipo_cambio' in self.datasets_procesados:
            df_tc = self.datasets_procesados['tipo_cambio']
            tc_actual = df_tc['dolar_principal'].dropna().iloc[-1]
            resumen_text += f"💱 TIPO DE CAMBIO:\n"
            resumen_text += f"• Valor actual: ${tc_actual:.2f}\n\n"
        
        resumen_text += "🎯 FACTORES CLAVE:\n"
        resumen_text += "• Estacionalidad agrícola\n"
        resumen_text += "• Volatilidad cambiaria\n"
        resumen_text += "• Inflación (IPC)\n"
        resumen_text += "• Condiciones climáticas"
        
        axes[2,2].text(0.05, 0.95, resumen_text, transform=axes[2,2].transAxes, 
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def generar_insights_predictivos(self):
        """Genera insights específicos para el modelo predictivo"""
        print("\n" + "="*60)
        print("🎯 INSIGHTS PARA MODELO PREDICTIVO DE ARCOR")
        print("="*60)
        
        insights = {}
        
        # Análisis de maíz
        if 'agrofy_maiz' in self.datasets_procesados:
            df_maiz = self.datasets_procesados['agrofy_maiz']
            
            print(f"\n🌽 ANÁLISIS PREDICTIVO DEL MAÍZ:")
            print("-" * 40)
            
            # Tendencia
            df_monthly = df_maiz.resample('M')['precio_numerico'].mean()
            df_monthly = df_monthly.dropna()
            tendencia = np.polyfit(range(len(df_monthly)), df_monthly.values, 1)[0]
            print(f"• Tendencia mensual: U$S {tendencia:.2f} por mes")
            
            # Volatilidad
            volatilidad = df_maiz['precio_numerico'].std()
            print(f"• Volatilidad histórica: U$S {volatilidad:.2f}")
            
            # Estacionalidad
            df_maiz['mes'] = df_maiz.index.month
            estacionalidad = df_maiz.groupby('mes')['precio_numerico'].mean()
            mes_mas_caro = estacionalidad.idxmax()
            mes_mas_barato = estacionalidad.idxmin()
            print(f"• Mes más caro: {mes_mas_caro} (U$S {estacionalidad[mes_mas_caro]:.2f})")
            print(f"• Mes más barato: {mes_mas_barato} (U$S {estacionalidad[mes_mas_barato]:.2f})")
            
            insights['maiz'] = {
                'tendencia_mensual': tendencia,
                'volatilidad': volatilidad,
                'mes_mas_caro': mes_mas_caro,
                'mes_mas_barato': mes_mas_barato
            }
        
        # Correlaciones cruzadas
        print(f"\n🔗 CORRELACIONES PARA PREDICCIÓN:")
        print("-" * 40)
        
        if 'agrofy_maiz' in self.datasets_procesados and 'tipo_cambio' in self.datasets_procesados:
            # Alinear datos
            df_maiz_m = self.datasets_procesados['agrofy_maiz'].resample('M')['precio_numerico'].mean()
            df_tc_m = self.datasets_procesados['tipo_cambio']['dolar_principal'].resample('M').mean()
            
            df_combined = pd.DataFrame({'maiz': df_maiz_m, 'tipo_cambio': df_tc_m}).dropna()
            
            if len(df_combined) > 0:
                corr_tc = df_combined['maiz'].corr(df_combined['tipo_cambio'])
                print(f"• Correlación Maíz-Tipo de Cambio: {corr_tc:.3f}")
                insights['correlacion_tipo_cambio'] = corr_tc
        
        if 'agrofy_maiz' in self.datasets_procesados and 'ipc' in self.datasets_procesados:
            # Correlación con IPC
            df_maiz_m = self.datasets_procesados['agrofy_maiz'].resample('M')['precio_numerico'].mean()
            df_ipc_m = self.datasets_procesados['ipc']['ipc_ng_nacional'].resample('M').mean()
            
            df_combined_ipc = pd.DataFrame({'maiz': df_maiz_m, 'ipc': df_ipc_m}).dropna()
            
            if len(df_combined_ipc) > 0:
                corr_ipc = df_combined_ipc['maiz'].corr(df_combined_ipc['ipc'])
                print(f"• Correlación Maíz-IPC: {corr_ipc:.3f}")
                insights['correlacion_ipc'] = corr_ipc
        
        return insights
    
    def ejecutar_eda_completo_mejorado(self):
        """Ejecuta el análisis exploratorio completo mejorado"""
        print("🌽 ANÁLISIS EXPLORATORIO - PREDICCIÓN PRECIOS MAÍZ ARCOR")
        print("=" * 80)
        
        # Cargar y procesar datos
        self.cargar_datos()
        self.procesar_datasets()
        
        self.analisis_datasets()
        df_maiz = self.analisis_maiz_detallado()
        
        # Crear visualizaciones mejoradas
        self.crear_visualizaciones_mejoradas()
        
        # Generar insights predictivos
        insights = self.generar_insights_predictivos()
        
        print(f"\n✅ EDA MEJORADO COMPLETADO EXITOSAMENTE")
        print(f"💡 {len(self.datasets)} datasets cargados")
        print(f"🔧 {len(self.datasets_procesados)} datasets procesados")
        
        return insights, df_maiz
