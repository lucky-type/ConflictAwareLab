import { useState, useMemo } from 'react';
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid, Legend, ReferenceLine } from 'recharts';
import { ChevronDown, ChevronRight, TrendingUp, Shield, Zap, BarChart3, GitCompare, Activity, Maximize2 } from 'lucide-react';
import type { ExperimentProgress } from '../../services/api';
import ZoomableChartModal from './ZoomableChartModal';

interface MetricsChartsProps {
  progressHistory: ExperimentProgress[];
  safetyEnabled: boolean;
  useEpisodes?: boolean; // For evaluation mode: show episodes instead of steps on X-axis
}

interface ChartSection {
  id: string;
  title: string;
  icon: any;
  color: string;
}

const CHART_SECTIONS: ChartSection[] = [
  { id: 'performance', title: 'Performance Metrics', icon: TrendingUp, color: 'cyan' },
  { id: 'safety', title: 'Safety / Risk Metrics', icon: Shield, color: 'rose' },
  { id: 'lagrangian', title: 'Lagrangian Dynamics', icon: Zap, color: 'amber' },
  { id: 'residual', title: 'Residual Learning', icon: GitCompare, color: 'purple' },
  { id: 'cars', title: 'CARS K Arbiter', icon: Activity, color: 'sky' },  // NEW
  { id: 'variance', title: 'Variance / Stability', icon: BarChart3, color: 'emerald' },
  { id: 'tradeoff', title: 'Trade-off Analysis', icon: GitCompare, color: 'violet' },
  { id: 'advanced', title: 'Advanced Metrics', icon: Activity, color: 'indigo' },
];

export default function MetricsCharts({ progressHistory, safetyEnabled, useEpisodes = false }: MetricsChartsProps) {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(['performance', 'safety', 'lagrangian', 'residual', 'cars', 'variance', 'tradeoff', 'advanced'])
  );

  const toggleSection = (sectionId: string) => {
    setExpandedSections(prev => {
      const newSet = new Set(prev);
      if (newSet.has(sectionId)) {
        newSet.delete(sectionId);
      } else {
        newSet.add(sectionId);
      }
      return newSet;
    });
  };

  // Prepare chart data
  const chartData = useMemo(() => {
    return progressHistory.map(p => ({
      step: p.step,
      episode: p.metrics.episode_count || 0,

      // Performance
      reward_raw: p.metrics.reward_raw_mean || 0,
      success_rate: p.metrics.success_rate || 0,
      crash_rate: p.metrics.crash_rate || 0,
      timeout_rate: p.metrics.timeout_rate || 0,
      ep_length: p.metrics.mean_ep_length || 0,

      // Safety
      cost_mean: p.metrics.cost_mean || 0,
      violation_rate: p.metrics.violation_rate || 0,
      near_miss_mean: p.metrics.near_miss_mean || 0,
      danger_time_mean: p.metrics.danger_time_mean || 0,

      // Lagrangian
      lambda: p.metrics.lambda || 0,
      reward_shaped: p.metrics.reward_shaped_mean || 0,
      epsilon: p.metrics.epsilon || 0,
      warmup_target_lambda: p.metrics.warmup_target_lambda || null,
      warmup_progress: p.metrics.warmup_progress || null,

      // Variance
      reward_std: p.metrics.reward_raw_std || 0,
      cost_std: p.metrics.cost_std || 0,
      ep_length_std: p.metrics.ep_length_std || 0,

      // Advanced
      policy_loss: p.metrics.policy_loss,
      value_loss: p.metrics.value_loss,
      actor_loss: p.metrics.actor_loss,
      critic_loss: p.metrics.critic_loss,
      entropy: p.metrics.entropy,
      learning_rate: p.metrics.learning_rate,

      // Residual Learning
      residual_correction: p.metrics.residual_correction_magnitude || 0,
      residual_base: p.metrics.residual_base_magnitude || 0,
      residual_ratio: p.metrics.residual_intervention_ratio || 0,

      // NEW: Advanced Residual Metrics
      residual_contribution: p.metrics.residual_contribution_mean || 0,
      conflict_mean: p.metrics.conflict_mean || 0,
      conflict_std: p.metrics.conflict_std || 0,
      intervention_rate: p.metrics.intervention_rate || 0,
      effective_k_mean: p.metrics.effective_k_mean || 0,

      // CARS K Components
      k_conf: p.metrics.k_conf_mean || 1,
      k_risk: p.metrics.k_risk_mean || 1,
    }));
  }, [progressHistory]);

  // Calculate steps-to-safety metric
  const stepsToSafety = useMemo(() => {
    if (chartData.length === 0) return null;

    const epsilon = chartData[chartData.length - 1]?.epsilon || 0;
    const windowSize = 50;

    for (let i = windowSize; i < chartData.length; i++) {
      const window = chartData.slice(i - windowSize, i);
      const avgCost = window.reduce((sum, d) => sum + d.cost_mean, 0) / windowSize;

      // Check if the next 50 episodes also maintain cost <= epsilon (stable)
      if (avgCost <= epsilon && i + windowSize < chartData.length) {
        const futureWindow = chartData.slice(i, i + windowSize);
        const futureAvgCost = futureWindow.reduce((sum, d) => sum + d.cost_mean, 0) / windowSize;
        if (futureAvgCost <= epsilon) {
          return chartData[i].step;
        }
      }
    }
    return null;
  }, [chartData]);

  const getLatestMetric = (key: keyof typeof chartData[0]) => {
    if (chartData.length === 0) return 0;
    return chartData[chartData.length - 1][key] || 0;
  };

  const epsilon = chartData.length > 0 ? chartData[chartData.length - 1].epsilon : 0;

  const hasAdvancedMetrics = chartData.some(d =>
    d.policy_loss !== undefined ||
    d.value_loss !== undefined ||
    d.actor_loss !== undefined ||
    d.critic_loss !== undefined ||
    d.entropy !== undefined ||
    d.learning_rate !== undefined
  );

  const hasResidualMetrics = chartData.some(d => d.residual_correction > 0 || d.residual_base > 0 || d.residual_ratio > 0 || d.residual_contribution > 0);

  // Check if CARS data is available (any K component != 1.0 or effective_k != 0)
  const hasCARSMetrics = chartData.some(d =>
    d.effective_k_mean > 0 || d.k_conf !== 1 || d.k_risk !== 1
  );

  // Check if any safety data exists
  const hasSafetyData = chartData.some(d =>
    d.cost_mean > 0 ||
    d.violation_rate > 0
  );

  // Show safety section if explicitly enabled OR if safety data exists
  const shouldShowSafety = safetyEnabled || hasSafetyData;

  if (progressHistory.length === 0) {
    return (
      <div className="text-center text-notion-text-secondary py-8">
        No metrics data available yet. Training will start collecting metrics soon.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* A) Performance Metrics */}
      <SectionCard
        section={CHART_SECTIONS[0]}
        expanded={expandedSections.has('performance')}
        onToggle={() => toggleSection('performance')}
      >
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <MetricDisplay label="Mean Reward (Raw)" value={getLatestMetric('reward_raw').toFixed(2)} color="cyan" />
          <MetricDisplay label="Success Rate" value={`${getLatestMetric('success_rate').toFixed(1)}%`} color="emerald" />
          <MetricDisplay label="Crash Rate" value={`${getLatestMetric('crash_rate').toFixed(1)}%`} color="rose" />
          <MetricDisplay label="Timeout Rate" value={`${getLatestMetric('timeout_rate').toFixed(1)}%`} color="orange" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ChartContainer
            title="Raw Reward Over Time"
            chartConfig={{
              data: chartData,
              xAxisKey: useEpisodes ? "episode" : "step",
              lines: [{ dataKey: "reward_raw", stroke: "#22d3ee" }]
            }}
          >
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                  labelStyle={{ color: '#374151' }}
                />
                <Line type="monotone" dataKey="reward_raw" stroke="#22d3ee" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>

          <ChartContainer
            title="Success, Crash, & Timeout Rates"
            chartConfig={{
              data: chartData,
              xAxisKey: useEpisodes ? "episode" : "step",
              lines: [
                { dataKey: "success_rate", stroke: "#34d399", name: "Success %" },
                { dataKey: "crash_rate", stroke: "#f87171", name: "Crash %" },
                { dataKey: "timeout_rate", stroke: "#fb923c", name: "Timeout %" }
              ]
            }}
          >
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                  labelStyle={{ color: '#374151' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="success_rate" stroke="#34d399" strokeWidth={2} dot={false} name="Success %" />
                <Line type="monotone" dataKey="crash_rate" stroke="#f87171" strokeWidth={2} dot={false} name="Crash %" />
                <Line type="monotone" dataKey="timeout_rate" stroke="#fb923c" strokeWidth={2} dot={false} name="Timeout %" />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        </div>
      </SectionCard>

      {/* B) Safety / Risk Metrics */}
      {shouldShowSafety && (
        <SectionCard
          section={CHART_SECTIONS[1]}
          expanded={expandedSections.has('safety')}
          onToggle={() => toggleSection('safety')}
        >
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
            <MetricDisplay label="Mean Episode Cost" value={getLatestMetric('cost_mean').toFixed(4)} color="rose" />
            <MetricDisplay label="Violation Rate" value={`${getLatestMetric('violation_rate').toFixed(1)}%`} color="orange" />
            <MetricDisplay label="Near-Misses (avg)" value={getLatestMetric('near_miss_mean').toFixed(4)} color="yellow" />
            <MetricDisplay label="Danger Time (avg)" value={getLatestMetric('danger_time_mean').toFixed(4)} color="yellow" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ChartContainer
              title="Episode Cost vs Risk Budget (ε)"
              chartConfig={{
                data: chartData,
                xAxisKey: useEpisodes ? "episode" : "step",
                lines: [{ dataKey: "cost_mean", stroke: "#f87171", name: "Mean Cost" }],
                referenceLines: [{ y: epsilon, stroke: "#fbbf24", label: "ε" }]
              }}
            >
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                  <YAxis stroke="#94a3b8" fontSize={11} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                    labelStyle={{ color: '#374151' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <ReferenceLine y={epsilon} stroke="#fbbf24" strokeDasharray="5 5" label={{ value: 'ε', fill: '#fbbf24', fontSize: 12 }} />
                  <Line type="monotone" dataKey="cost_mean" stroke="#f87171" strokeWidth={2} dot={false} name="Mean Cost" />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer
              title="Safety Violations"
              chartConfig={{
                data: chartData,
                xAxisKey: useEpisodes ? "episode" : "step",
                lines: [
                  { dataKey: "violation_rate", stroke: "#fb923c", name: "Violation %" },
                ]
              }}
            >
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                  <YAxis stroke="#94a3b8" fontSize={11} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                    labelStyle={{ color: '#374151' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Line type="monotone" dataKey="violation_rate" stroke="#fb923c" strokeWidth={2} dot={false} name="Violation %" />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>
        </SectionCard>
      )}

      {/* C) Lagrangian Dynamics - Always show (lambda is always tracked) */}
      <SectionCard
        section={CHART_SECTIONS[2]}
        expanded={expandedSections.has('lagrangian')}
        onToggle={() => toggleSection('lagrangian')}
      >
        <div className="grid grid-cols-2 gap-4 mb-4">
          <MetricDisplay label="λ (Lambda)" value={getLatestMetric('lambda').toFixed(4)} color="amber" />
          <MetricDisplay label="Shaped Reward Mean" value={getLatestMetric('reward_shaped').toFixed(2)} color="orange" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ChartContainer
            title="λ (Lagrangian Multiplier) Evolution"
            chartConfig={{
              data: chartData,
              xAxisKey: useEpisodes ? "episode" : "step",
              lines: [
                { dataKey: "lambda", stroke: "#fbbf24", name: "λ (current)" },
                { dataKey: "warmup_target_lambda", stroke: "#f97316", name: "Target λ (warmup)", strokeDasharray: "5 5" }
              ]
            }}
          >
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                  labelStyle={{ color: '#374151' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="lambda" stroke="#fbbf24" strokeWidth={2} dot={false} name="λ (current)" />
                <Line
                  type="monotone"
                  dataKey="warmup_target_lambda"
                  stroke="#f97316"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                  name="Target λ (warmup)"
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>

          <ChartContainer
            title="Raw vs Shaped Reward"
            chartConfig={{
              data: chartData,
              xAxisKey: useEpisodes ? "episode" : "step",
              lines: [
                { dataKey: "reward_raw", stroke: "#22d3ee", name: "Raw (r)" },
                { dataKey: "reward_shaped", stroke: "#fb923c", name: "Shaped (r')" }
              ]
            }}
          >
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                  labelStyle={{ color: '#374151' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="reward_raw" stroke="#22d3ee" strokeWidth={2} dot={false} name="Raw (r)" />
                <Line type="monotone" dataKey="reward_shaped" stroke="#fb923c" strokeWidth={2} dot={false} name="Shaped (r')" />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        </div>
      </SectionCard>

      {/* D) Residual Learning Dynamics */}
      {hasResidualMetrics && (
        <SectionCard
          section={CHART_SECTIONS[3]}
          expanded={expandedSections.has('residual')}
          onToggle={() => toggleSection('residual')}
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <MetricDisplay label="Conflict Mean (Cos Sim)" value={getLatestMetric('conflict_mean').toFixed(3)} color="rose" />
            <MetricDisplay label="Contribution Ratio" value={getLatestMetric('residual_contribution').toFixed(3)} color="purple" />
            <MetricDisplay label="Intervention Rate" value={`${getLatestMetric('intervention_rate').toFixed(1)}%`} color="pink" />
            <MetricDisplay label="Effective K (mean)" value={getLatestMetric('effective_k_mean').toFixed(4)} color="cyan" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ChartContainer
              title="Policy Conflict (Cosine Similarity)"
              chartConfig={{
                data: chartData,
                xAxisKey: useEpisodes ? "episode" : "step",
                lines: [
                  { dataKey: "conflict_mean", stroke: "#f43f5e", name: "Cos Sim Mean" },
                  { dataKey: "conflict_std", stroke: "#fb923c", name: "Std Dev", strokeDasharray: "3 3", strokeWidth: 1 }
                ],
                yAxisDomain: [-1, 1],
                referenceLines: [{ y: 0, stroke: "#475569", strokeDasharray: "3 3" }]
              }}
            >
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                  <YAxis stroke="#94a3b8" fontSize={11} domain={[-1, 1]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                    labelStyle={{ color: '#374151' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="conflict_mean" stroke="#f43f5e" strokeWidth={2} dot={false} name="Cos Sim Mean" />
                  <Line type="monotone" dataKey="conflict_std" stroke="#fb923c" strokeWidth={1} strokeDasharray="3 3" dot={false} name="Std Dev" />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer
              title="Residual Contribution Ratio (0-1)"
              chartConfig={{
                data: chartData,
                xAxisKey: useEpisodes ? "episode" : "step",
                lines: [
                  { dataKey: "residual_contribution", stroke: "#a855f7", name: "Contribution" },
                  { dataKey: "intervention_rate", stroke: "#ec4899", name: "Intervention Rate / 100" }
                ],
                yAxisDomain: [0, 1]
              }}
            >
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                  <YAxis stroke="#94a3b8" fontSize={11} domain={[0, 1]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                    labelStyle={{ color: '#374151' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Line type="monotone" dataKey="residual_contribution" stroke="#a855f7" strokeWidth={2} dot={false} name="Contribution" />
                  <Line type="monotone" dataKey="intervention_rate" stroke="#ec4899" strokeWidth={2} dot={false} name="Intervention Rate / 100" />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>

          {/* NEW: Effective K Factor chart */}
          <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ChartContainer
              title="Effective K-Factor (Adaptive)"
              chartConfig={{
                data: chartData,
                xAxisKey: useEpisodes ? "episode" : "step",
                lines: [{ dataKey: "effective_k_mean", stroke: "#22d3ee", name: "Effective K" }]
              }}
            >
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                  <YAxis stroke="#94a3b8" fontSize={11} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                    labelStyle={{ color: '#374151' }}
                  />
                  <Line type="monotone" dataKey="effective_k_mean" stroke="#22d3ee" strokeWidth={2} dot={false} name="Effective K" />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>

          <div className="mt-4 rounded-lg bg-purple-50 border border-purple-200 px-4 py-3">
            <p className="text-xs text-purple-700 leading-relaxed">
              <strong>Residual Learning:</strong> Policy Conflict measures alignment between base and residual policies (1.0 = aligned, -1.0 = opposed).
              <strong> Adaptive K:</strong> When enabled, K-factor is reduced when residual opposes base action (conflict prevention).
            </p>
          </div>
        </SectionCard>
      )}

      {/* NEW: CARS K Arbiter Section */}
      {hasCARSMetrics && (
        <SectionCard
          section={CHART_SECTIONS[4]}
          expanded={expandedSections.has('cars')}
          onToggle={() => toggleSection('cars')}
        >
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
            <MetricDisplay label="K effective" value={getLatestMetric('effective_k_mean').toFixed(4)} color="cyan" />
            <MetricDisplay label="K conf" value={getLatestMetric('k_conf').toFixed(3)} color="rose" />
            <MetricDisplay label="K risk" value={getLatestMetric('k_risk').toFixed(3)} color="amber" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ChartContainer title="Adaptive K (Effective)">
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                  <YAxis stroke="#94a3b8" fontSize={11} domain={[0, 'auto']} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                    labelStyle={{ color: '#374151' }}
                  />
                  <Line type="monotone" dataKey="effective_k_mean" stroke="#22d3ee" strokeWidth={2} dot={false} name="Effective K" />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer
              title="CARS K Components"
              chartConfig={{
                data: chartData,
                xAxisKey: useEpisodes ? "episode" : "step",
                lines: [
                  { dataKey: "k_conf", stroke: "#f43f5e", name: "K_conf (conflict)" },
                  { dataKey: "k_risk", stroke: "#f59e0b", name: "K_risk (λ)" },
                ],
                yAxisDomain: [0, 1.1],
                referenceLines: [{ y: 1, stroke: "#475569", strokeDasharray: "3 3" }]
              }}
            >
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                  <YAxis stroke="#94a3b8" fontSize={11} domain={[0, 1.1]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                    labelStyle={{ color: '#374151' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <ReferenceLine y={1} stroke="#475569" strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="k_conf" stroke="#f43f5e" strokeWidth={2} dot={false} name="K_conf (conflict)" />
                  <Line type="monotone" dataKey="k_risk" stroke="#f59e0b" strokeWidth={2} dot={false} name="K_risk (λ)" />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>

          <div className="mt-4 rounded-lg bg-blue-50 border border-blue-200 px-4 py-3">
            <p className="text-xs text-blue-700 leading-relaxed">
              <strong>CARS (Conflict-Aware Residual Scaling):</strong> K = K_base × K_conf × K_risk.
              Each component defaults to 1.0 if not applicable. Lower values = more conservative residual corrections.
            </p>
          </div>
        </SectionCard>
      )}

      {/* E) Variance / Stability */}
      <SectionCard
        section={CHART_SECTIONS[5]}
        expanded={expandedSections.has('variance')}
        onToggle={() => toggleSection('variance')}
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ChartContainer
            title="Reward Stability (Std Dev)"
            chartConfig={{
              data: chartData,
              xAxisKey: useEpisodes ? "episode" : "step",
              lines: [{ dataKey: "reward_std", stroke: "#34d399", name: "Reward Std Dev" }]
            }}
          >
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                  labelStyle={{ color: '#374151' }}
                />
                <Line type="monotone" dataKey="reward_std" stroke="#34d399" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>

          {safetyEnabled && (
            <ChartContainer
              title="Cost Stability (Std Dev)"
              chartConfig={{
                data: chartData,
                xAxisKey: useEpisodes ? "episode" : "step",
                lines: [{ dataKey: "cost_std", stroke: "#f87171", name: "Cost Std Dev" }]
              }}
            >
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                  <YAxis stroke="#94a3b8" fontSize={11} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                    labelStyle={{ color: '#374151' }}
                  />
                  <Line type="monotone" dataKey="cost_std" stroke="#f87171" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          )}
        </div>
      </SectionCard>

      {/* F) Trade-off Analysis */}
      {safetyEnabled && (
        <SectionCard
          section={CHART_SECTIONS[6]}
          expanded={expandedSections.has('tradeoff')}
          onToggle={() => toggleSection('tradeoff')}
        >
          <div className="grid grid-cols-1 gap-4 mb-4">
            {stepsToSafety !== null && (
              <MetricDisplay
                label="Steps to Safety (moving avg cost ≤ ε stable)"
                value={stepsToSafety.toLocaleString()}
                color="emerald"
              />
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ChartContainer
              title="Reward vs Cost Over Time"
              chartConfig={{
                data: chartData,
                xAxisKey: useEpisodes ? "episode" : "step",
                lines: [
                  { dataKey: "reward_raw", stroke: "#a855f7", name: "Reward", yAxisId: "left" },
                  { dataKey: "cost_mean", stroke: "#f87171", name: "Cost", yAxisId: "right" }
                ],
                dualYAxis: true,
                referenceLines: [{ y: epsilon, stroke: "#fbbf24", strokeDasharray: "5 5", label: "ε" }]
              }}
            >
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                  <YAxis yAxisId="left" stroke="#94a3b8" fontSize={11} />
                  <YAxis yAxisId="right" orientation="right" stroke="#94a3b8" fontSize={11} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                    labelStyle={{ color: '#374151' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <ReferenceLine yAxisId="right" y={epsilon} stroke="#fbbf24" strokeDasharray="5 5" label={{ value: 'ε', fill: '#fbbf24', fontSize: 12 }} />
                  <Line yAxisId="left" type="monotone" dataKey="reward_raw" stroke="#a855f7" strokeWidth={2} dot={false} name="Reward" />
                  <Line yAxisId="right" type="monotone" dataKey="cost_mean" stroke="#f87171" strokeWidth={2} dot={false} name="Cost" />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer
              title="Success Rate vs Cost Over Time"
              chartConfig={{
                data: chartData,
                xAxisKey: useEpisodes ? "episode" : "step",
                lines: [
                  { dataKey: "success_rate", stroke: "#22d3ee", name: "Success %", yAxisId: "left" },
                  { dataKey: "cost_mean", stroke: "#f87171", name: "Cost", yAxisId: "right" }
                ],
                dualYAxis: true,
                referenceLines: [{ y: epsilon, stroke: "#fbbf24", strokeDasharray: "5 5", label: "ε" }]
              }}
            >
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                  <YAxis yAxisId="left" stroke="#94a3b8" fontSize={11} />
                  <YAxis yAxisId="right" orientation="right" stroke="#94a3b8" fontSize={11} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                    labelStyle={{ color: '#374151' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <ReferenceLine yAxisId="right" y={epsilon} stroke="#fbbf24" strokeDasharray="5 5" label={{ value: 'ε', fill: '#fbbf24', fontSize: 12 }} />
                  <Line yAxisId="left" type="monotone" dataKey="success_rate" stroke="#22d3ee" strokeWidth={2} dot={false} name="Success %" />
                  <Line yAxisId="right" type="monotone" dataKey="cost_mean" stroke="#f87171" strokeWidth={2} dot={false} name="Cost" />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>
        </SectionCard>
      )}

      {/* G) Advanced Metrics (Algorithm-specific) - Hidden in evaluation mode */}
      {!useEpisodes && (
        <SectionCard
          section={CHART_SECTIONS[7]}
          expanded={expandedSections.has('advanced')}
          onToggle={() => toggleSection('advanced')}
        >
          {chartData.some(d => d.policy_loss !== undefined || d.value_loss !== undefined || d.actor_loss !== undefined || d.critic_loss !== undefined) ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* On-policy algorithms (PPO, A2C) */}
              {chartData.some(d => d.policy_loss !== undefined) && (
                <ChartContainer
                  title="Policy Loss (PPO/A2C)"
                  chartConfig={{
                    data: chartData,
                    xAxisKey: useEpisodes ? "episode" : "step",
                    lines: [{ dataKey: "policy_loss", stroke: "#818cf8", name: "Policy Loss" }]
                  }}
                >
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                      <YAxis stroke="#94a3b8" fontSize={11} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                        labelStyle={{ color: '#374151' }}
                      />
                      <Line type="monotone" dataKey="policy_loss" stroke="#818cf8" strokeWidth={2} dot={false} name="Policy Loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartContainer>
              )}

              {chartData.some(d => d.value_loss !== undefined) && (
                <ChartContainer
                  title="Value Loss (PPO/A2C)"
                  chartConfig={{
                    data: chartData,
                    xAxisKey: useEpisodes ? "episode" : "step",
                    lines: [{ dataKey: "value_loss", stroke: "#c084fc", name: "Value Loss" }]
                  }}
                >
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                      <YAxis stroke="#94a3b8" fontSize={11} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                        labelStyle={{ color: '#374151' }}
                      />
                      <Line type="monotone" dataKey="value_loss" stroke="#c084fc" strokeWidth={2} dot={false} name="Value Loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartContainer>
              )}

              {/* Off-policy algorithms (SAC, TD3, DDPG) */}
              {chartData.some(d => d.actor_loss !== undefined) && (
                <ChartContainer
                  title="Actor Loss (SAC/TD3/DDPG)"
                  chartConfig={{
                    data: chartData,
                    xAxisKey: useEpisodes ? "episode" : "step",
                    lines: [{ dataKey: "actor_loss", stroke: "#06b6d4", name: "Actor Loss" }]
                  }}
                >
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                      <YAxis stroke="#94a3b8" fontSize={11} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                        labelStyle={{ color: '#374151' }}
                      />
                      <Line type="monotone" dataKey="actor_loss" stroke="#06b6d4" strokeWidth={2} dot={false} name="Actor Loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartContainer>
              )}

              {chartData.some(d => d.critic_loss !== undefined) && (
                <ChartContainer
                  title="Critic Loss (SAC/TD3/DDPG)"
                  chartConfig={{
                    data: chartData,
                    xAxisKey: useEpisodes ? "episode" : "step",
                    lines: [{ dataKey: "critic_loss", stroke: "#f59e0b", name: "Critic Loss" }]
                  }}
                >
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                      <YAxis stroke="#94a3b8" fontSize={11} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                        labelStyle={{ color: '#374151' }}
                      />
                      <Line type="monotone" dataKey="critic_loss" stroke="#f59e0b" strokeWidth={2} dot={false} name="Critic Loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartContainer>
              )}

              {/* Entropy (PPO, A2C) */}
              {chartData.some(d => d.entropy !== undefined) && (
                <ChartContainer
                  title="Entropy Loss (PPO/A2C)"
                  chartConfig={{
                    data: chartData,
                    xAxisKey: useEpisodes ? "episode" : "step",
                    lines: [{ dataKey: "entropy", stroke: "#10b981", name: "Entropy Loss" }]
                  }}
                >
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                      <YAxis stroke="#94a3b8" fontSize={11} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                        labelStyle={{ color: '#374151' }}
                      />
                      <Line type="monotone" dataKey="entropy" stroke="#10b981" strokeWidth={2} dot={false} name="Entropy Loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartContainer>
              )}

              {/* Learning Rate (All algorithms) */}
              {chartData.some(d => d.learning_rate !== undefined) && (
                <ChartContainer
                  title="Learning Rate"
                  chartConfig={{
                    data: chartData,
                    xAxisKey: useEpisodes ? "episode" : "step",
                    lines: [{ dataKey: "learning_rate", stroke: "#ec4899", name: "Learning Rate" }]
                  }}
                >
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey={useEpisodes ? "episode" : "step"} stroke="#94a3b8" fontSize={11} />
                      <YAxis stroke="#94a3b8" fontSize={11} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                        labelStyle={{ color: '#374151' }}
                      />
                      <Line type="monotone" dataKey="learning_rate" stroke="#ec4899" strokeWidth={2} dot={false} name="Learning Rate" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartContainer>
              )}
            </div>
          ) : (
            <div className="text-center text-notion-text-secondary py-4 text-sm">
              Algorithm-specific metrics not available yet
            </div>
          )}
        </SectionCard>
      )}
    </div>
  );
}

function SectionCard({ section, expanded, onToggle, children }: {
  section: ChartSection;
  expanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  const Icon = section.icon;
  const ChevronIcon = expanded ? ChevronDown : ChevronRight;

  return (
    <div className="rounded-xl border border-notion-border bg-white overflow-hidden shadow-sm">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-4 hover:bg-notion-hover transition"
      >
        <div className="flex items-center gap-3">
          <div className={`rounded-lg bg-${section.color}-50 p-2`}>
            <Icon size={18} className={`text-${section.color}-500`} />
          </div>
          <h3 className="text-sm font-semibold text-notion-text">{section.title}</h3>
        </div>
        <ChevronIcon size={20} className="text-notion-text-secondary" />
      </button>

      {expanded && (
        <div className="p-4 pt-0 border-t border-notion-border">
          {children}
        </div>
      )}
    </div>
  );
}

interface ChartConfig {
  data: any[];
  xAxisKey: string;
  lines: {
    dataKey: string;
    stroke: string;
    name?: string;
    strokeWidth?: number;
    strokeDasharray?: string;
    yAxisId?: string;
  }[];
  yAxisDomain?: [number | 'auto', number | 'auto'];
  referenceLines?: {
    y: number;
    stroke: string;
    strokeDasharray?: string;
    label?: string;
  }[];
  dualYAxis?: boolean;
}

function ChartContainer({
  title,
  children,
  chartConfig
}: {
  title: string;
  children: React.ReactNode;
  chartConfig?: ChartConfig;
}) {
  const [showZoomModal, setShowZoomModal] = useState(false);

  return (
    <>
      <div className="rounded-lg border border-notion-border bg-notion-light-gray p-3">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-xs font-medium text-notion-text-secondary">{title}</h4>
          {chartConfig && (
            <button
              onClick={() => setShowZoomModal(true)}
              className="p-1 rounded hover:bg-notion-hover text-notion-text-tertiary hover:text-notion-blue transition"
              title="Expand chart with zoom controls"
            >
              <Maximize2 size={14} />
            </button>
          )}
        </div>
        {children}
      </div>

      {chartConfig && (
        <ZoomableChartModal
          isOpen={showZoomModal}
          onClose={() => setShowZoomModal(false)}
          title={title}
          data={chartConfig.data}
          lines={chartConfig.lines}
          xAxisKey={chartConfig.xAxisKey}
          yAxisDomain={chartConfig.yAxisDomain}
          referenceLines={chartConfig.referenceLines}
          dualYAxis={chartConfig.dualYAxis}
        />
      )}
    </>
  );
}


function MetricDisplay({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="rounded-lg border border-notion-border bg-notion-light-gray p-3">
      <div className="text-xs text-notion-text-secondary mb-1">{label}</div>
      <div className={`text-lg font-semibold text-${color}-600`}>{value}</div>
    </div>
  );
}
