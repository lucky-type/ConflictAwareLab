import { X } from 'lucide-react';
import { Card, CardHeader } from '../common/Card';
import { useEffect, useState } from 'react';
import { api, type ModelSnapshot, type ExperimentMetric } from '../../services/api';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  zoomPlugin
);

interface Props {
  open: boolean;
  onClose: () => void;
  snapshots: ModelSnapshot[];
}

const MODEL_COLORS = [
  '#6366f1', // Indigo
  '#f59e0b', // Amber
  '#10b981', // Emerald
  '#ec4899', // Pink
  '#8b5cf6', // Violet
  '#14b8a6', // Teal
  '#f97316', // Orange
  '#06b6d4', // Cyan
];

export default function CompareModelsModal({ open, onClose, snapshots }: Props) {
  const [metricsData, setMetricsData] = useState<Record<number, ExperimentMetric[]>>({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (open && snapshots.length > 0) {
      loadMetrics();
    }
  }, [open, snapshots]);

  const loadMetrics = async () => {
    setLoading(true);
    try {
      const metricsMap: Record<number, ExperimentMetric[]> = {};
      
      for (const snapshot of snapshots) {
        const metrics = await api.getExperimentMetrics(snapshot.experiment_id);
        // Filter metrics up to snapshot iteration
        const filteredMetrics = metrics.filter(m => m.step <= snapshot.iteration);
        metricsMap[snapshot.id] = filteredMetrics;
      }
      
      setMetricsData(metricsMap);
    } catch (err) {
      console.error('Failed to load metrics:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!open) return null;

  // Extract unique metric types from all snapshots (excluding 'step')
  const getMetricTypes = () => {
    const types = new Set<string>();
    Object.values(metricsData).forEach(metrics => {
      metrics.forEach(metric => {
        if (metric.values && typeof metric.values === 'object') {
          Object.keys(metric.values).forEach(key => {
            if (key !== 'step') types.add(key);
          });
        }
      });
    });
    return Array.from(types);
  };

  const metricTypes = getMetricTypes();

  // Create chart data for a specific metric type
  const createChartData = (metricType: string) => {
    const datasets = snapshots.map((snapshot, index) => {
      const metrics = metricsData[snapshot.id] || [];
      const data = metrics
        .filter(m => m.values && m.values[metricType] !== undefined)
        .map(m => ({
          x: m.step,
          y: m.values[metricType],
        }));

      return {
        label: `Snapshot #${snapshot.id} (Exp ${snapshot.experiment_id}, Iter ${snapshot.iteration})`,
        data,
        borderColor: MODEL_COLORS[index % MODEL_COLORS.length],
        backgroundColor: MODEL_COLORS[index % MODEL_COLORS.length] + '40',
        borderWidth: 2,
        pointRadius: 1,
        tension: 0.1,
      };
    });

    return {
      datasets,
    };
  };

  // Calculate summary statistics for comparison table
  const calculateStats = () => {
    return snapshots.map((snapshot, index) => {
      const metrics = metricsData[snapshot.id] || [];
      const stats: Record<string, { avg: number; max: number; min: number; final: number }> = {};

      metricTypes.forEach(type => {
        const values = metrics
          .filter(m => m.values && m.values[type] !== undefined)
          .map(m => m.values[type]);

        if (values.length > 0) {
          stats[type] = {
            avg: values.reduce((a, b) => a + b, 0) / values.length,
            max: Math.max(...values),
            min: Math.min(...values),
            final: values[values.length - 1] || 0,
          };
        }
      });

      return {
        snapshot,
        stats,
        color: MODEL_COLORS[index % MODEL_COLORS.length],
      };
    });
  };

  const statsData = calculateStats();

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: {
          color: '#374151',
          font: {
            size: 11,
          },
        },
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
      },
      zoom: {
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true,
          },
          mode: 'xy' as const,
        },
        pan: {
          enabled: true,
          mode: 'xy' as const,
        },
        limits: {
          x: { min: 'original' as const, max: 'original' as const },
          y: { min: 'original' as const, max: 'original' as const },
        },
      },
    },
    scales: {
      x: {
        type: 'linear' as const,
        title: {
          display: true,
          text: 'Step',
          color: '#888',
        },
        ticks: {
          color: '#888',
        },
        grid: {
          color: 'rgba(136, 136, 136, 0.1)',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Value',
          color: '#888',
        },
        ticks: {
          color: '#888',
        },
        grid: {
          color: 'rgba(136, 136, 136, 0.1)',
        },
      },
    },
  };

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm"
    >
      <div className="w-full max-w-7xl max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
        <Card className="p-6">
          <div className="flex items-start justify-between mb-6">
            <CardHeader title="Model Comparison" />
            <button onClick={onClose} className="text-notion-text-secondary hover:text-notion-text">
              <X size={20} />
            </button>
          </div>

          {loading ? (
            <div className="text-center py-12 text-notion-text-secondary">
              Loading metrics...
            </div>
          ) : (
            <>
              {/* Model Legend */}
              <div className="mb-6 p-4 bg-notion-light-gray rounded-xl border border-notion-border">
                <h3 className="text-sm font-semibold text-notion-text mb-3">Models</h3>
                <div className="grid grid-cols-2 gap-2">
                  {snapshots.map((snapshot, index) => (
                    <div key={snapshot.id} className="flex items-center gap-2">
                      <div
                        className="w-4 h-4 rounded"
                        style={{ backgroundColor: MODEL_COLORS[index % MODEL_COLORS.length] }}
                      />
                      <span className="text-sm text-notion-text">
                        Snapshot #{snapshot.id} (Exp {snapshot.experiment_id}, Iter {snapshot.iteration})
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Charts */}
              <div className="space-y-6 mb-6">
                <h3 className="text-lg font-semibold text-notion-text">Metrics Over Time</h3>
                {metricTypes.map(metricType => (
                  <div key={metricType} className="bg-notion-light-gray rounded-xl border border-notion-border p-4">
                    <h4 className="text-sm font-semibold text-notion-text mb-3 capitalize">
                      {metricType.replace(/_/g, ' ')}
                    </h4>
                    <div style={{ height: '300px' }}>
                      <Line data={createChartData(metricType)} options={chartOptions} />
                    </div>
                  </div>
                ))}
              </div>

              {/* Comparison Table */}
              <div className="bg-notion-light-gray rounded-xl border border-notion-border p-4">
                <h3 className="text-lg font-semibold text-notion-text mb-4">Summary Statistics</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-700">
                        <th className="text-left py-2 px-3 text-notion-text-secondary font-semibold">Model</th>
                        <th className="text-left py-2 px-3 text-notion-text-secondary font-semibold">Metric</th>
                        <th className="text-right py-2 px-3 text-notion-text-secondary font-semibold">Average</th>
                        <th className="text-right py-2 px-3 text-notion-text-secondary font-semibold">Max</th>
                        <th className="text-right py-2 px-3 text-notion-text-secondary font-semibold">Min</th>
                        <th className="text-right py-2 px-3 text-notion-text-secondary font-semibold">Final</th>
                      </tr>
                    </thead>
                    <tbody>
                      {statsData.map((model, modelIndex) => (
                        metricTypes.map((metricType, metricIndex) => (
                          <tr 
                            key={`${model.snapshot.id}-${metricType}`}
                            className="border-b border-notion-border/50 hover:bg-notion-hover/30"
                          >
                            {metricIndex === 0 && (
                              <td 
                                className="py-2 px-3 font-medium"
                                rowSpan={metricTypes.length}
                                style={{ color: model.color }}
                              >
                                <div className="flex items-center gap-2">
                                  <div
                                    className="w-3 h-3 rounded"
                                    style={{ backgroundColor: model.color }}
                                  />
                                  Snapshot #{model.snapshot.id}
                                </div>
                              </td>
                            )}
                            <td className="py-2 px-3 text-notion-text capitalize">
                              {metricType.replace(/_/g, ' ')}
                            </td>
                            {model.stats[metricType] ? (
                              <>
                                <td className="py-2 px-3 text-right" style={{ color: model.color }}>
                                  {model.stats[metricType].avg.toFixed(2)}
                                </td>
                                <td className="py-2 px-3 text-right" style={{ color: model.color }}>
                                  {model.stats[metricType].max.toFixed(2)}
                                </td>
                                <td className="py-2 px-3 text-right" style={{ color: model.color }}>
                                  {model.stats[metricType].min.toFixed(2)}
                                </td>
                                <td className="py-2 px-3 text-right" style={{ color: model.color }}>
                                  {model.stats[metricType].final.toFixed(2)}
                                </td>
                              </>
                            ) : (
                              <>
                                <td className="py-2 px-3 text-right text-slate-600">N/A</td>
                                <td className="py-2 px-3 text-right text-slate-600">N/A</td>
                                <td className="py-2 px-3 text-right text-slate-600">N/A</td>
                                <td className="py-2 px-3 text-right text-slate-600">N/A</td>
                              </>
                            )}
                          </tr>
                        ))
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </Card>
      </div>
    </div>
  );
}
