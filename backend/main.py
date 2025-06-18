from MLExtension import EnhancedRobustCornPipeline
from eda import EDAMaizArcorMejorado
from visualizaciones_arcor import graficar_resultados

dataset_paths = {
    'tipo_cambio': 'datos/tipos-de-cambio-historicos.csv',
    'maiz_datos': 'datos/maiz_ecc.csv',
    'ipc': 'datos/indice-precios-al-consumidor-nivel-general-base-diciembre-2016-mensual.csv',
    'clima': 'datos/Estadísticas normales Datos abiertos 1991-2020.xlsx',
    'agrofy_precios': 'datos/series-historicas-pizarra.csv'
}

eda_mejorado = EDAMaizArcorMejorado(dataset_paths)
insights, df_maiz = eda_mejorado.ejecutar_eda_completo_mejorado()

EnhancedRobustCornPipeline = EnhancedRobustCornPipeline(eda_mejorado)
EnhancedRobustCornPipeline.run_complete_analysis()

graficar_resultados(EnhancedRobustCornPipeline)
