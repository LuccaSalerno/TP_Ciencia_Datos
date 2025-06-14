import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import {
  TrendingUp, TrendingDown, DollarSign, BarChart3
} from 'lucide-react';

const ArcorPredictiveSystem = () => {
  const [historicalData, setHistoricalData] = useState({ maiz: [] });
  const [predictions, setPredictions] = useState([]);
  const [selectedTab, setSelectedTab] = useState('grafico');
  const [modelEvaluations, setModelEvaluations] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const pastRes = await fetch("http://localhost:8000/api/predicciones/maiz");
        const pastData = await pastRes.json();

        const futureRes = await fetch("http://localhost:8000/api/prediccion?horizonte=3");
        const futureData = await futureRes.json();

        const evalRes = await fetch("http://localhost:8000/api/evaluacion_modelos");
        const evalData = await evalRes.json();
        setModelEvaluations(evalData);

        const lastDate = new Date(pastData[pastData.length - 1].fecha);

        const past = pastData.map(d => {
          const fecha = new Date(d.fecha);
          return {
            fecha: d.fecha,
            mes: fecha.toLocaleDateString('es-AR', { month: 'short', year: '2-digit' }),
            precio: d.precio,
            prediccion_completa: d.prediccion,
            origen: "pasado"
          };
        });

        const future = futureData.predicciones.map((p, i) => {
          const fecha = new Date(lastDate);
          fecha.setMonth(fecha.getMonth() + i + 1);

          return {
            fecha: fecha.toISOString().split("T")[0],
            mes: fecha.toLocaleDateString('es-AR', { month: 'short', year: '2-digit' }),
            precio: null,
            prediccion_completa: p.precio_predicho,
            origen: "futuro"
          };
        });

        const merged = [...past, ...future];

        setHistoricalData({ maiz: merged });

        setPredictions(past.slice(-6).map(p => ({
          mes: p.mes,
          maiz: p.prediccion_completa,
          confianza: 90
        })));
      } catch (error) {
        console.error("❌ Error cargando datos:", error);
      }
    };

    fetchData();
  }, []);

  const currentData = historicalData.maiz;
  const latestPrice = currentData.findLast(d => d.precio != null)?.precio || 0;
  const previousPrice = currentData.slice(-2).find(d => d.precio != null)?.precio || 0;
  const priceChange = latestPrice - previousPrice;
  const priceChangePercent = previousPrice > 0 ? (priceChange / previousPrice) * 100 : 0;

  const MetricCard = ({ title, value, changePercent, icon: Icon, color }) => (
    <div className="bg-white rounded-lg shadow-md p-6 border-l-4" style={{ borderLeftColor: color }}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">${value?.toLocaleString('es-AR') || '0'}</p>
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

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Sistema Predictivo Arcor</h1>
        <p className="text-sm text-gray-600">Precio real, predicción pasada y predicción futura</p>
      </header>

      {/* Tabs */}
      <div className="flex space-x-4 mb-6">
        <button onClick={() => setSelectedTab('grafico')} className={`px-4 py-2 rounded ${selectedTab === 'grafico' ? 'bg-red-600 text-white' : 'bg-white text-red-600 border'}`}>Gráfico</button>
        <button onClick={() => setSelectedTab('evaluacion')} className={`px-4 py-2 rounded ${selectedTab === 'evaluacion' ? 'bg-blue-600 text-white' : 'bg-white text-blue-600 border'}`}>Evaluación de Modelos</button>
      </div>

      {/* Tab: Gráfico */}
      {selectedTab === 'grafico' && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <MetricCard
              title="Precio Actual"
              value={latestPrice}
              changePercent={priceChangePercent}
              icon={DollarSign}
              color="#dc2626"
            />
            <MetricCard
              title="Última Predicción"
              value={predictions[predictions.length - 1]?.maiz}
              changePercent={((predictions[predictions.length - 1]?.maiz - latestPrice) / latestPrice) * 100}
              icon={TrendingUp}
              color="#059669"
            />
            <MetricCard
              title="Confianza del Modelo"
              value={predictions[0]?.confianza}
              changePercent={0}
              icon={BarChart3}
              color="#7c3aed"
            />
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Precio vs Predicción</h2>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={currentData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="mes" />
                <YAxis />
                <Tooltip />
                <Legend />

                <Line type="monotone" dataKey="precio" stroke="#dc2626" strokeWidth={2} dot={false} name="Precio Real" />
                <Line type="monotone" dataKey={(d) => d.origen === "pasado" ? d.prediccion_completa : null} stroke="#059669" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Predicción Modelo (pasado)" isAnimationActive={false} />
                <Line type="monotone" dataKey={(d) => d.origen === "futuro" ? d.prediccion_completa : null} stroke="#0ea5e9" strokeWidth={2} strokeDasharray="3 3" dot={{ r: 2 }} name="Predicción Futura" isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {/* Tab: Evaluación */}
      {selectedTab === 'evaluacion' && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Evaluación de Modelos</h2>
          <table className="w-full text-left text-sm">
            <thead className="text-gray-700 border-b">
              <tr>
                <th className="py-2">Modelo</th>
                <th className="py-2">MAE</th>
                <th className="py-2">MAPE</th>
                <th className="py-2">R² Score</th>
              </tr>
            </thead>
            <tbody>
              {modelEvaluations.map((m, i) => (
                <tr key={i} className="border-b">
                  <td className="py-2 font-medium">{m.nombre}</td>
                  <td className="py-2">{m.mae.toFixed(2)}</td>
                  <td className="py-2">{m.mape.toFixed(2)}%</td>
                  <td className="py-2">{m.r2.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default ArcorPredictiveSystem;
