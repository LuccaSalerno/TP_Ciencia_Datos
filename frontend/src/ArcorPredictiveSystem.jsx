import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar, ScatterChart, Scatter } from 'recharts';
import { TrendingUp, TrendingDown, AlertTriangle, DollarSign, Calendar, Cloud, Globe, BarChart3, Settings, Bell } from 'lucide-react';

// Datos simulados para el prototipo
const generateHistoricalData = (commodity, months = 24) => {
  const data = [];
  const basePrice = commodity === 'azucar' ? 800 : commodity === 'maiz' ? 300 : 450;
  const volatility = commodity === 'azucar' ? 0.15 : commodity === 'maiz' ? 0.12 : 0.18;
  
  for (let i = 0; i < months; i++) {
    const date = new Date();
    date.setMonth(date.getMonth() - months + i);
    
    const trend = Math.sin(i * 0.5) * 0.1;
    const seasonal = Math.sin(i * 0.8) * 0.08;
    const noise = (Math.random() - 0.5) * volatility;
    
    const price = basePrice * (1 + trend + seasonal + noise);
    
    data.push({
      fecha: date.toISOString().split('T')[0],
      mes: date.toLocaleDateString('es-AR', { month: 'short', year: '2-digit' }),
      precio: Math.round(price),
      prediccion: i > months - 6 ? Math.round(price * (1 + (Math.random() - 0.5) * 0.05)) : null,
      intervalo_superior: i > months - 6 ? Math.round(price * 1.08) : null,
      intervalo_inferior: i > months - 6 ? Math.round(price * 0.92) : null
    });
  }
  return data;
};

const generatePredictionData = (months = 6) => {
  const predictions = [];
  const baseDate = new Date();
  
  for (let i = 1; i <= months; i++) {
    const date = new Date();
    date.setMonth(date.getMonth() + i);
    
    predictions.push({
      mes: date.toLocaleDateString('es-AR', { month: 'short', year: '2-digit' }),
      azucar: Math.round(850 + (Math.random() - 0.5) * 100),
      maiz: Math.round(320 + (Math.random() - 0.5) * 40),
      leche: Math.round(480 + (Math.random() - 0.5) * 60),
      confianza: 85 + Math.random() * 10
    });
  }
  return predictions;
};

const scenarioData = [
  { escenario: 'Base', azucar: 850, maiz: 320, leche: 480, probabilidad: 40 },
  { escenario: 'Sequía Severa', azucar: 920, maiz: 380, leche: 520, probabilidad: 15 },
  { escenario: 'Devaluación 20%', azucar: 1020, maiz: 384, leche: 576, probabilidad: 20 },
  { escenario: 'Condiciones Óptimas', azucar: 780, maiz: 290, leche: 420, probabilidad: 25 }
];

const riskFactors = [
  { factor: 'Clima', impacto: 'Alto', probabilidad: 30, descripcion: 'Sequía en zona productiva' },
  { factor: 'Tipo de Cambio', impacto: 'Medio', probabilidad: 45, descripcion: 'Volatilidad cambiaria' },
  { factor: 'Políticas Comerciales', impacto: 'Medio', probabilidad: 25, descripcion: 'Restricciones a exportación' },
  { factor: 'Precios Internacionales', impacto: 'Alto', probabilidad: 35, descripcion: 'Volatilidad mercados globales' }
];

const ArcorPredictiveSystem = () => {
  const [selectedCommodity, setSelectedCommodity] = useState('azucar');
  const [selectedTab, setSelectedTab] = useState('predicciones');
  const [historicalData, setHistoricalData] = useState({});
  const [predictions, setPredictions] = useState([]);
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    // Simular carga de datos
    setHistoricalData({
      azucar: generateHistoricalData('azucar'),
      maiz: generateHistoricalData('maiz'),
      leche: generateHistoricalData('leche')
    });
    setPredictions(generatePredictionData());
    
    // Simular alertas
    setAlerts([
      { id: 1, tipo: 'warning', mensaje: 'Precio del azúcar muestra tendencia alcista (+5.2%)', timestamp: '2 horas' },
      { id: 2, tipo: 'info', mensaje: 'Recomendación: Considerar compra de maíz en próximas 2 semanas', timestamp: '1 día' },
      { id: 3, tipo: 'alert', mensaje: 'Alerta climática: Sequía prevista en zona productiva', timestamp: '3 horas' }
    ]);
  }, []);

  const commodityNames = {
    azucar: 'Azúcar',
    maiz: 'Maíz',
    leche: 'Leche en Polvo'
  };

  const currentData = historicalData[selectedCommodity] || [];
  const latestPrice = currentData.length > 0 ? currentData[currentData.length - 1]?.precio : 0;
  const previousPrice = currentData.length > 1 ? currentData[currentData.length - 2]?.precio : 0;
  const priceChange = latestPrice - previousPrice;
  const priceChangePercent = previousPrice > 0 ? ((priceChange / previousPrice) * 100) : 0;

  const MetricCard = ({ title, value, change, changePercent, icon: Icon, color }) => (
    <div className="bg-white rounded-lg shadow-md p-6 border-l-4" style={{ borderLeftColor: color }}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">
            ${value?.toLocaleString('es-AR') || '0'}
          </p>
          <div className="flex items-center mt-1">
            {changePercent > 0 ? (
              <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
            )}
            <span className={`text-sm font-medium ${changePercent > 0 ? 'text-green-600' : 'text-red-600'}`}>
              {changePercent > 0 ? '+' : ''}{changePercent?.toFixed(1)}%
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
                <p className="text-sm text-gray-600">Predicción de Precios de Materias Primas</p>
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
              { id: 'riesgo', name: 'Análisis de Riesgo', icon: AlertTriangle },
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
        {selectedTab === 'predicciones' && (
          <div className="space-y-6">
            {/* Commodity Selector */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Seleccionar Materia Prima</h2>
              <div className="flex space-x-4">
                {Object.entries(commodityNames).map(([key, name]) => (
                  <button
                    key={key}
                    onClick={() => setSelectedCommodity(key)}
                    className={`px-4 py-2 rounded-lg font-medium ${
                      selectedCommodity === key
                        ? 'bg-red-600 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    {name}
                  </button>
                ))}
              </div>
            </div>

            {/* Current Price Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <MetricCard
                title={`Precio Actual - ${commodityNames[selectedCommodity]}`}
                value={latestPrice}
                change={priceChange}
                changePercent={priceChangePercent}
                icon={DollarSign}
                color="#dc2626"
              />
              <MetricCard
                title="Predicción 30 días"
                value={predictions[0]?.[selectedCommodity]}
                change={predictions[0]?.[selectedCommodity] - latestPrice}
                changePercent={((predictions[0]?.[selectedCommodity] - latestPrice) / latestPrice) * 100}
                icon={TrendingUp}
                color="#059669"
              />
              <MetricCard
                title="Confianza del Modelo"
                value={Math.round(predictions[0]?.confianza || 85)}
                changePercent={0}
                icon={BarChart3}
                color="#7c3aed"
              />
            </div>

            {/* Historical and Prediction Chart */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Evolución y Predicción de Precios - {commodityNames[selectedCommodity]}
              </h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={currentData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="mes" />
                  <YAxis />
                  <Tooltip 
                    formatter={(value, name) => [
                      `$${value?.toLocaleString('es-AR')}`, 
                      name === 'precio' ? 'Precio Real' : 'Predicción'
                    ]}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="precio" 
                    stroke="#dc2626" 
                    strokeWidth={2}
                    name="Precio Histórico"
                    connectNulls={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="prediccion" 
                    stroke="#059669" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="Predicción"
                    connectNulls={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="intervalo_superior" 
                    stroke="#94a3b8" 
                    strokeWidth={1}
                    strokeDasharray="2 2"
                    name="Intervalo Superior"
                    connectNulls={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="intervalo_inferior" 
                    stroke="#94a3b8" 
                    strokeWidth={1}
                    strokeDasharray="2 2"
                    name="Intervalo Inferior"
                    connectNulls={false}
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
                        Azúcar ($/kg)
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Maíz ($/kg)
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Leche en Polvo ($/kg)
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Confianza
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {predictions.map((pred, index) => (
                      <tr key={index}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {pred.mes}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          ${pred.azucar.toLocaleString('es-AR')}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          ${pred.maiz.toLocaleString('es-AR')}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          ${pred.leche.toLocaleString('es-AR')}
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
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'escenarios' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Análisis de Escenarios</h2>
              <p className="text-gray-600 mb-6">
                Simulación de precios bajo diferentes condiciones macroeconómicas y climáticas
              </p>
              
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={scenarioData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="escenario" />
                  <YAxis />
                  <Tooltip 
                    formatter={(value, name) => [
                      `$${value?.toLocaleString('es-AR')}`, 
                      name === 'azucar' ? 'Azúcar' : name === 'maiz' ? 'Maíz' : 'Leche en Polvo'
                    ]}
                  />
                  <Legend />
                  <Bar dataKey="azucar" fill="#dc2626" name="Azúcar" />
                  <Bar dataKey="maiz" fill="#059669" name="Maíz" />
                  <Bar dataKey="leche" fill="#7c3aed" name="Leche en Polvo" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {scenarioData.map((scenario, index) => (
                <div key={index} className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">{scenario.escenario}</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Probabilidad:</span>
                      <span className="font-medium">{scenario.probabilidad}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Impacto Azúcar:</span>
                      <span className="font-medium">${scenario.azucar.toLocaleString('es-AR')}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Impacto Maíz:</span>
                      <span className="font-medium">${scenario.maiz.toLocaleString('es-AR')}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Impacto Leche:</span>
                      <span className="font-medium">${scenario.leche.toLocaleString('es-AR')}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {selectedTab === 'riesgo' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Matriz de Factores de Riesgo</h2>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Factor de Riesgo
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Impacto
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Probabilidad
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Descripción
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {riskFactors.map((risk, index) => (
                      <tr key={index}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {risk.factor}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${ 
                            risk.impacto === 'Alto' ? 'bg-red-100 text-red-800' : 
                            risk.impacto === 'Medio' ? 'bg-yellow-100 text-yellow-800' : 
                            'bg-green-100 text-green-800'
                          }`}>
                            {risk.impacto}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {risk.probabilidad}%
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-900">
                          {risk.descripcion}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Value at Risk (VaR)</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <p className="text-2xl font-bold text-red-600">7.5%</p>
                  <p className="text-sm text-gray-600">VaR 95% - 1 mes</p>
                  <p className="text-xs text-gray-500">Pérdida máxima esperada</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-orange-600">12.3%</p>
                  <p className="text-sm text-gray-600">VaR 95% - 3 meses</p>
                  <p className="text-xs text-gray-500">Pérdida máxima esperada</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-yellow-600">18.7%</p>
                  <p className="text-sm text-gray-600">VaR 95% - 6 meses</p>
                  <p className="text-xs text-gray-500">Pérdida máxima esperada</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'alertas' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Alertas Activas</h2>
              <div className="space-y-4">
                {alerts.map((alert) => (
                  <AlertCard key={alert.id} alert={alert} />
                ))}
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
