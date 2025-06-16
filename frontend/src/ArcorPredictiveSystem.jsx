import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, AlertTriangle, DollarSign, Calendar, BarChart3, Settings, Bell } from 'lucide-react';

const ArcorPredictiveSystem = () => {
  const [historicalData, setHistoricalData] = useState({ maiz: [] });
  const [predictions, setPredictions] = useState([]);
  const [selectedTab, setSelectedTab] = useState('predicciones');
  const [selectedCommodity, setSelectedCommodity] = useState('maiz');
  const [modelEvaluations, setModelEvaluations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [escenarios, setEscenarios] = useState([]);
  const [volatilidad, setVolatilidad] = useState(null);
  const [precioBase, setPrecioBase] = useState(null);
  const [diagnostico, setDiagnostico] = useState(null);


  const commodityNames = {
    azucar: 'Azúcar',
    maiz: 'Maíz',
    leche: 'Leche en Polvo'
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Datos históricos y predicciones pasadas
        const pastRes = await fetch("http://localhost:8000/api/predicciones/maiz");
        const pastData = await pastRes.json();

        // Predicciones futuras
        const futureRes = await fetch("http://localhost:8000/api/prediccion?horizonte=6");
        const futureData = await futureRes.json();

        // Evaluación de modelos
        const evalRes = await fetch("http://localhost:8000/api/evaluacion_modelos");
        const evalData = await evalRes.json();
        setModelEvaluations(evalData);

        // Escenarios simulados
        const escenariosRes = await fetch("http://localhost:8000/api/escenarios");
        const escenariosData = await escenariosRes.json();
        setEscenarios(escenariosData.escenarios);
        setVolatilidad(escenariosData.volatilidad_historica);
        setPrecioBase(escenariosData.precio_base);

        // Diagnóstico del modelo
        const diagRes = await fetch("http://localhost:8000/api/diagnosticos");
        const diagData = await diagRes.json();
        setDiagnostico(diagData);



        const lastDate = new Date(pastData[pastData.length - 1].fecha);

        // Procesamiento de datos pasados
        const past = pastData.map(d => {
          const fecha = new Date(d.fecha);
          return {
            fecha: d.fecha,
            mes: fecha.toLocaleDateString('es-AR', { month: 'short', year: '2-digit' }),
            precio: d.precio,
            prediccion_pasada: d.prediccion,
            origen: "pasado"
          };
        });

        // Procesamiento de predicciones futuras
        const future = futureData.predicciones.map((p, i) => {
          const fecha = new Date(lastDate);
          fecha.setMonth(fecha.getMonth() + i + 1);

          return {
            fecha: fecha.toISOString().split("T")[0],
            mes: p.mes,
            precio: null,
            prediccion_futura: p.precio_predicho,
            intervalo_superior: p.limite_superior,
            intervalo_inferior: p.limite_inferior,
            confianza: p.confianza,
            origen: "futuro"
          };
        });

        const merged = [...past, ...future];
        setHistoricalData({ maiz: merged });

        // Generar predicciones para la tabla
        const predictionTable = future.slice(0, 6).map(f => ({
          mes: f.mes,
          maiz: f.prediccion_futura,
          confianza: f.confianza
        }));
        setPredictions(predictionTable);

        // Simular alertas basadas en datos reales
        const latestPrice = past[past.length - 1]?.precio || 0;
        const previousPrice = past[past.length - 2]?.precio || 0;
        const priceChange = ((latestPrice - previousPrice) / previousPrice) * 100;
        
        const simulatedAlerts = [];
        if (Math.abs(priceChange) > 3) {
          simulatedAlerts.push({
            id: 1,
            tipo: priceChange > 0 ? 'warning' : 'alert',
            mensaje: `Precio del maíz ${priceChange > 0 ? 'aumentó' : 'disminuyó'} ${Math.abs(priceChange).toFixed(1)}% en el último período`,
            timestamp: '2 horas'
          });
        }
        
        simulatedAlerts.push({
          id: 2,
          tipo: 'info',
          mensaje: 'Modelo actualizado con nuevos datos de mercado',
          timestamp: '1 día'
        });

        setAlerts(simulatedAlerts);
        
      } catch (error) {
        console.error("❌ Error cargando datos:", error);
        setError("Error al cargar los datos del servidor");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const currentData = historicalData.maiz;
  const latestPrice = currentData.findLast(d => d.precio != null)?.precio || 0;
  const previousPrice = currentData.slice(-2).find(d => d.precio != null)?.precio || 0;
  const priceChange = latestPrice - previousPrice;
  const priceChangePercent = previousPrice > 0 ? (priceChange / previousPrice) * 100 : 0;

  const MetricCard = ({ title, value, changePercent, icon: Icon, color, suffix = "" }) => (
    <div className="bg-white rounded-lg shadow-md p-6 border-l-4" style={{ borderLeftColor: color }}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">
            {suffix ? `${Math.round(value)}${suffix}` : `$${value?.toLocaleString('es-AR') || '0'}`}
          </p>
          <div className="flex items-center mt-1">
            {changePercent > 0 ? (
              <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
            ) : changePercent < 0 ? (
              <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
            ) : null}
            <span className={`text-sm font-medium ${
              changePercent > 0 ? 'text-green-600' : 
              changePercent < 0 ? 'text-red-600' : 'text-gray-600'
            }`}>
              {changePercent !== 0 && (changePercent > 0 ? '+' : '')}{changePercent?.toFixed(1)}%
            </span>
          </div>
        </div>
        <Icon className="w-8 h-8" style={{ color }} />
      </div>
    </div>
  );

  const AlertCard = ({ alert }) => {
    const getAlertColor = (tipo) => {
      switch (tipo) {
        case 'warning': return 'border-yellow-500 bg-yellow-50';
        case 'alert': return 'border-red-500 bg-red-50';
        default: return 'border-blue-500 bg-blue-50';
      }
    };

    const getAlertIcon = (tipo) => {
      switch (tipo) {
        case 'warning': return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
        case 'alert': return <AlertTriangle className="w-5 h-5 text-red-600" />;
        default: return <Bell className="w-5 h-5 text-blue-600" />;
      }
    };

    return (
      <div className={`border rounded-lg p-4 ${getAlertColor(alert.tipo)}`}>
        <div className="flex items-start space-x-3">
          {getAlertIcon(alert.tipo)}
          <div className="flex-1">
            <p className="text-sm font-medium text-gray-900">{alert.mensaje}</p>
            <p className="text-xs text-gray-500 mt-1">Hace {alert.timestamp}</p>
          </div>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-red-600 rounded-lg flex items-center justify-center mb-4 mx-auto">
            <span className="text-white font-bold text-xl">A</span>
          </div>
          <p className="text-gray-600">Cargando datos...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <p className="text-gray-900 font-semibold mb-2">Error al cargar datos</p>
          <p className="text-gray-600">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-red-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">A</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Sistema Predictivo Arcor</h1>
                <p className="text-sm text-gray-600">Predicción de Precios de Maíz</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <p className="text-sm text-gray-600">Última actualización</p>
                <p className="text-sm font-medium">{new Date().toLocaleString('es-AR')}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8">
            {[
              { id: 'predicciones', name: 'Predicciones', icon: TrendingUp },
              { id: 'escenarios', name: 'Escenarios', icon: BarChart3 },
              { id: 'evaluacion', name: 'Evaluación de Modelos', icon: AlertTriangle },
              { id: 'diagnostico', name: 'Diagnóstico', icon: Settings },

              // { id: 'riesgo', name: 'Análisis de Riesgo', icon: AlertTriangle },
              { id: 'alertas', name: 'Alertas', icon: Bell }
            ].map(({ id, name, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setSelectedTab(id)}
                className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                  selectedTab === id
                    ? 'border-red-500 text-red-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{name}</span>
              </button>
            ))}
          </nav>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Commodity Selector */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Seleccionar Materia Prima</h2>
          <div className="flex space-x-4">
            {Object.entries(commodityNames).map(([key, name]) => (
              <button
                key={key}
                onClick={() => key === 'maiz' && setSelectedCommodity(key)}
                disabled={key !== 'maiz'}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  selectedCommodity === key
                    ? 'bg-red-600 text-white'
                    : key === 'maiz'
                      ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                      : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                }`}
              >
                {name}
                {key !== 'maiz' && (
                  <span className="ml-2 text-xs">(Próximamente)</span>
                )}
              </button>
            ))}
          </div>
        </div>

        {selectedTab === 'predicciones' && (
          <div className="space-y-6">
            {/* Current Price Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <MetricCard
                title="Precio Actual - Maíz"
                value={latestPrice}
                changePercent={priceChangePercent}
                icon={DollarSign}
                color="#dc2626"
              />
              <MetricCard
                title="Predicción Próximo Mes"
                value={predictions[0]?.maiz}
                changePercent={predictions[0]?.maiz ? ((predictions[0].maiz - latestPrice) / latestPrice) * 100 : 0}
                icon={TrendingUp}
                color="#059669"
              />
              <MetricCard
                title="Confianza del Modelo"
                value={predictions[0]?.confianza || 85}
                changePercent={0}
                icon={BarChart3}
                color="#7c3aed"
                suffix="%"
              />
            </div>

            {/* Historical and Prediction Chart */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Evolución y Predicción de Precios - Maíz
              </h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={currentData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="mes" />
                  <YAxis />
                  <Tooltip 
                    formatter={(value, name) => {
                      if (!value) return ['N/A', name];
                      const formatName = {
                        'precio': 'Precio Real',
                        'prediccion_pasada': 'Predicción Pasada',
                        'prediccion_futura': 'Predicción Futura',
                        'intervalo_superior': 'Intervalo Superior',
                        'intervalo_inferior': 'Intervalo Inferior'
                      };
                      return [`$${value?.toLocaleString('es-AR')}`, formatName[name] || name];
                    }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="precio" 
                    stroke="#dc2626" 
                    strokeWidth={2}
                    name="Precio Real"
                    connectNulls={false}
                    dot={{ r: 3 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="prediccion_pasada" 
                    stroke="#059669" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="Predicción Pasada"
                    connectNulls={false}
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="prediccion_futura" 
                    stroke="#0ea5e9" 
                    strokeWidth={2}
                    strokeDasharray="3 3"
                    name="Predicción Futura"
                    connectNulls={false}
                    dot={{ r: 4 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="intervalo_superior" 
                    stroke="#94a3b8" 
                    strokeWidth={1}
                    strokeDasharray="2 2"
                    name="Intervalo Superior"
                    connectNulls={false}
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="intervalo_inferior" 
                    stroke="#94a3b8" 
                    strokeWidth={1}
                    strokeDasharray="2 2"
                    name="Intervalo Inferior"
                    connectNulls={false}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Predictions Table */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Predicciones Detalladas</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Mes
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Precio Predicho ($/kg)
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Confianza
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Cambio vs Actual
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {predictions.map((pred, index) => {
                      const changeVsActual = ((pred.maiz - latestPrice) / latestPrice) * 100;
                      return (
                        <tr key={index}>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                            {pred.mes}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            ${pred.maiz?.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            <div className="flex items-center">
                              <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                                <div 
                                  className="bg-green-600 h-2 rounded-full" 
                                  style={{ width: `${pred.confianza}%` }}
                                ></div>
                              </div>
                              <span>{Math.round(pred.confianza)}%</span>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm">
                            <span className={`font-medium ${
                              changeVsActual > 0 ? 'text-green-600' : 
                              changeVsActual < 0 ? 'text-red-600' : 'text-gray-600'
                            }`}>
                              {changeVsActual > 0 ? '+' : ''}{changeVsActual.toFixed(1)}%
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'escenarios' && (
          <div className="bg-white rounded-lg shadow-md p-6 space-y-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Análisis de Escenarios</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {escenarios.map((e, index) => (
                <div key={index} className="border rounded-lg p-4 shadow-sm bg-gray-50">
                  <h3 className="text-md font-semibold text-gray-800 mb-2">{e.escenario}</h3>
                  <p className="text-sm text-gray-600 mb-2">{e.descripcion}</p>
                  <div className="space-y-1 text-sm text-gray-700">
                    <p>📈 Precio Estimado: <strong>${e.maiz.toLocaleString('es-AR')}</strong></p>
                    <p>🔽 Límite Inferior: ${e.limite_inferior.toLocaleString('es-AR')}</p>
                    <p>🔼 Límite Superior: ${e.limite_superior.toLocaleString('es-AR')}</p>
                    <p>📊 Probabilidad: {e.probabilidad}%</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 text-sm text-gray-600">
              <p>📌 Precio base actual: <strong>${precioBase?.toLocaleString('es-AR')}</strong></p>
              <p>📉 Volatilidad histórica del precio: <strong>{volatilidad}%</strong></p>
            </div>
          </div>
        )}


        {selectedTab === 'riesgo' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Análisis de Riesgo</h2>
            <p className="text-gray-600 mb-6">
              Evaluación de riesgos asociados a las fluctuaciones de precios de materias primas.
            </p>
            <div className="text-center py-12">
              <AlertTriangle className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500 text-lg">Módulo en desarrollo</p>
            </div>
          </div>
        )}

        {selectedTab === 'evaluacion' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Evaluación de Modelos</h2>
            <p className="text-gray-600 mb-6">
              Métricas de rendimiento de los modelos predictivos utilizados
            </p>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Modelo
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      MAE (Error Absoluto Medio)
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      MAPE (Error Porcentual Medio)
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      R² Score
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      RMSE (Raíz del Error Cuadrático Medio)
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      STD
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Escalado
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Estado
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {modelEvaluations.modelos
                  .sort((a, b) => {
                    if (a.es_mejor) return -1;
                    if (b.es_mejor) return 1;
                    return a.MAPE - b.MAPE;
                  })
                  .map((model, index) => (
                    <tr key={index}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {model.nombre}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {model.MAE?.toFixed(2)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {model.MAPE?.toFixed(2)}%
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        <span className={`font-medium ${
                          model.R2 > 0.8 ? 'text-green-600' : 
                          model.R2 > 0.6 ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {model.R2?.toFixed(3)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {model.RMSE?.toFixed(2)}%
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {model.pred_std?.toFixed(2)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {model.usa_escalado ? "Sí" : "No"}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          model.es_mejor
                            ? 'bg-green-100 text-green-800'
                            : model.R2 > 0.6
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-red-100 text-red-800'
                        }`}>
                          {model.es_mejor
                            ? '⭐ Mejor Modelo'
                            : model.R2 > 0.8
                              ? 'Excelente'
                              : model.R2 > 0.6
                                ? 'Bueno'
                                : 'Regular'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {selectedTab === 'diagnostico' && diagnostico && (
          <div className="bg-white rounded-lg shadow-md p-6 space-y-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Diagnóstico del Modelo</h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Datos */}
              <div className="bg-gray-50 rounded-lg p-4 shadow-sm">
                <h3 className="text-md font-semibold text-gray-700 mb-2">Datos</h3>
                <p><strong>Observaciones:</strong> {diagnostico.datos.total_observaciones}</p>
                <p><strong>Fechas:</strong> {diagnostico.datos.rango_fechas.inicio} → {diagnostico.datos.rango_fechas.fin}</p>
                <p><strong>Valores faltantes:</strong> {diagnostico.datos.valores_faltantes}</p>
                <p><strong>Precio promedio:</strong> ${diagnostico.datos.precio_promedio}</p>
                <p><strong>Volatilidad:</strong> {diagnostico.datos.precio_volatilidad}</p>
              </div>

              {/* Modelo */}
              <div className="bg-gray-50 rounded-lg p-4 shadow-sm">
                <h3 className="text-md font-semibold text-gray-700 mb-2">Modelo</h3>
                <p><strong>Nombre:</strong> {diagnostico.modelo.nombre}</p>
                <p><strong>Precisión (MAPE):</strong> {diagnostico.modelo.precision}%</p>
                <p><strong>R²:</strong> {diagnostico.modelo.r_cuadrado}</p>
                <p><strong>Error Promedio (MAE):</strong> {diagnostico.modelo.error_promedio}</p>
                <p><strong>Features usadas:</strong> {diagnostico.modelo.features_utilizadas}</p>
                <p><strong>Escalado:</strong> {diagnostico.modelo.usa_escalado ? "Sí" : "No"}</p>
              </div>

              {/* Rendimiento */}
              <div className="bg-gray-50 rounded-lg p-4 shadow-sm">
                <h3 className="text-md font-semibold text-gray-700 mb-2">Rendimiento</h3>
                <p><strong>Predicciones test:</strong> {diagnostico.rendimiento.predicciones_test}</p>
                <p><strong>Error std:</strong> {diagnostico.rendimiento.error_std}</p>
                <p><strong>Sesgo promedio:</strong> {diagnostico.rendimiento.sesgo_promedio}</p>
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'alertas' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Alertas Activas</h2>
              <div className="space-y-4">
                {alerts.length > 0 ? (
                  alerts.map((alert) => (
                    <AlertCard key={alert.id} alert={alert} />
                  ))
                ) : (
                  <div className="text-center py-8">
                    <Bell className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">No hay alertas activas</p>
                  </div>
                )}
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Configuración de Alertas</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Umbral de Variación de Precio (%)
                  </label>
                  <input 
                    type="number" 
                    defaultValue="5" 
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-red-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Frecuencia de Notificaciones
                  </label>
                  <select className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-red-500">
                    <option>Inmediata</option>
                    <option>Diaria</option>
                    <option>Semanal</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
        )}

        
      </div>
    </div>
  );
};

export default ArcorPredictiveSystem;