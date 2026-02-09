import { X, Activity, BarChart3 } from 'lucide-react';
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from 'recharts';
import { Card, CardHeader } from '../common/Card';
import SimulationViewer from '../simulation/SimulationViewer';
import MetricsCharts from '../charts/MetricsCharts';
import type { ExperimentProgress, SimulationFrame, ExperimentStatus, Experiment } from '../../services/api';
import { useEffect, useState } from 'react';
import { api } from '../../services/api';

interface MetricDataPoint {
  step: number;
  value: number;
}

interface MetricsHistory {
  mean_reward: MetricDataPoint[];
  mean_ep_length: MetricDataPoint[];
  [key: string]: MetricDataPoint[];
}

interface Props {
  open: boolean;
  onClose: (shouldReopen?: boolean) => void;
  experimentName: string;
  experimentType: 'Training' | 'Simulation' | 'Fine-Tuning' | 'Evaluation';
  experimentId?: number;
  experimentStatus?: ExperimentStatus;
  totalSteps?: number; // Total steps from experiment configuration
  maxStep?: number; // For snapshot viewing - limit metrics to this step
  progress?: ExperimentProgress;
  frame?: SimulationFrame;
  experiment?: Experiment; // Full experiment object for environment dimensions
}

export default function TrainingInfoModal({
  open,
  onClose,
  experimentName,
  experimentType,
  experimentId,
  experimentStatus,
  totalSteps,
  maxStep,
  progress,
  frame,
  experiment,
}: Props) {
  const [metricsHistory, setMetricsHistory] = useState<MetricsHistory>({
    mean_reward: [],
    mean_ep_length: [],
  });
  const [progressHistory, setProgressHistory] = useState<ExperimentProgress[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [showVisualization, setShowVisualization] = useState(true);

  // Stable state for cumulative evaluation metrics (prevents flickering from out-of-order WS messages)
  const [stableOutcomes, setStableOutcomes] = useState<{ successes: number; failures: number; timeouts: number }>({ successes: 0, failures: 0, timeouts: 0 });

  // Debug: Log experiment data
  useEffect(() => {
    console.log('TrainingInfoModal experiment:', experiment);
    console.log('TrainingInfoModal environment:', experiment?.environment);
  }, [experiment]);

  // Replay state
  const [isReplaying, setIsReplaying] = useState(false);
  const [replayFrame, setReplayFrame] = useState<SimulationFrame | null>(null);
  const [replayTrajectory, setReplayTrajectory] = useState<any[]>([]);
  const [currentReplayIndex, setCurrentReplayIndex] = useState(0);
  const [replaySpeed, setReplaySpeed] = useState(1.0); // 1x speed

  // Load historical metrics for completed/cancelled experiments
  useEffect(() => {
    if (open && experimentId && experimentStatus) {
      setIsLoadingHistory(true);

      console.log('[TrainingInfoModal] Loading historical data for experiment:', experimentId, 'status:', experimentStatus);

      // Load metrics
      api.getExperimentMetrics(experimentId)
        .then((metrics) => {
          console.log('[TrainingInfoModal] Loaded metrics:', metrics.length);

          const history: MetricsHistory = {
            mean_reward: [],
            mean_ep_length: [],
          };
          const progressHistoryData: ExperimentProgress[] = [];

          // Filter metrics by maxStep if provided (for snapshot viewing)
          const filteredMetrics = maxStep
            ? metrics.filter(m => m.step <= maxStep)
            : metrics;

          filteredMetrics.forEach((metric) => {
            const step = metric.step;
            const values = metric.values;

            // Build progress history for MetricsCharts
            progressHistoryData.push({
              experiment_id: experimentId,
              step: step,
              total_steps: totalSteps || 100000,
              metrics: values,
              status: experimentStatus || 'Unknown',
              timestamp: Date.now(),
            });

            // Add mean_reward
            if (values.mean_reward !== undefined) {
              history.mean_reward.push({ step, value: values.mean_reward });
            }

            // Add mean_ep_length
            if (values.mean_ep_length !== undefined) {
              history.mean_ep_length.push({ step, value: values.mean_ep_length });
            }

            // Add other metrics dynamically
            Object.keys(values).forEach((key) => {
              if (key !== 'mean_reward' && key !== 'mean_ep_length' && key !== 'step') {
                if (!history[key]) {
                  history[key] = [];
                }
                history[key].push({ step, value: values[key] });
              }
            });
          });


          setMetricsHistory(history);
          setProgressHistory(progressHistoryData);

          // Fix: Populate stable outcomes from the last metric point (for completed experiments)
          if (progressHistoryData.length > 0) {
            const lastMetric = progressHistoryData[progressHistoryData.length - 1].metrics;
            if (lastMetric) {
              setStableOutcomes({
                successes: lastMetric.total_successes ?? 0,
                failures: lastMetric.total_failures ?? 0,
                timeouts: lastMetric.total_timeouts ?? 0,
              });
            }
          }
        })
        .catch((error) => {
          console.error('[TrainingInfoModal] Failed to load historical data:', error);
        })
        .finally(() => {
          setIsLoadingHistory(false);
        });

      // Load trajectory for replay (both Training and Simulation)
      if (experimentId) {
        api.getExperimentTrajectory(experimentId)
          .then(data => {
            console.log('[TrainingInfoModal] Loaded trajectory:', data.trajectory?.length || 0, 'frames');
            setReplayTrajectory(data.trajectory || []);
          })
          .catch(err => {
            console.warn('No trajectory data available:', err);
          });
      }
    }
  }, [open, experimentId, experimentStatus, experimentType, maxStep]);

  // Update metrics history when new progress data arrives (for in-progress experiments)
  useEffect(() => {
    // For evaluation mode, check episode instead of step (step is 0 for evaluation)
    const hasProgress = progress && progress.metrics && (progress.step > 0 || (progress.episode !== undefined && progress.episode > 0));

    if (hasProgress) {
      // Use episode for evaluation mode, step for training mode
      const progressKey = experimentType === 'Evaluation' && progress.episode !== undefined ? progress.episode : progress.step;

      // Add to progress history for MetricsCharts
      setProgressHistory((prev) => {
        const lastPoint = prev[prev.length - 1];
        // Check both step and episode for deduplication
        const lastKey = experimentType === 'Evaluation' && lastPoint?.episode !== undefined ? lastPoint.episode : lastPoint?.step;
        if (!lastPoint || lastKey !== progressKey) {
          return [...prev, progress].slice(-200); // Keep last 200 points
        }
        return prev;
      });

      setMetricsHistory((prev) => {
        const updated = { ...prev };
        const stepKey = progressKey; // Use the same key as progressHistory

        // Add mean_reward
        if (progress.metrics.mean_reward !== undefined) {
          const lastPoint = updated.mean_reward[updated.mean_reward.length - 1];
          if (!lastPoint || lastPoint.step !== stepKey) {
            updated.mean_reward = [...updated.mean_reward, {
              step: stepKey,
              value: progress.metrics.mean_reward,
            }].slice(-100); // Keep last 100 points
          }
        }

        // Add mean_ep_length
        if (progress.metrics.mean_ep_length !== undefined) {
          const lastPoint = updated.mean_ep_length[updated.mean_ep_length.length - 1];
          if (!lastPoint || lastPoint.step !== stepKey) {
            updated.mean_ep_length = [...updated.mean_ep_length, {
              step: stepKey,
              value: progress.metrics.mean_ep_length,
            }].slice(-100);
          }
        }

        // Add other metrics dynamically
        Object.keys(progress.metrics).forEach((key) => {
          if (key !== 'mean_reward' && key !== 'mean_ep_length' && key !== 'step') {
            const value = (progress.metrics as Record<string, any>)[key];
            if (typeof value === 'number') {
              if (!updated[key]) {
                updated[key] = [];
              }
              const lastPoint = updated[key][updated[key].length - 1];
              if (!lastPoint || lastPoint.step !== stepKey) {
                updated[key] = [...updated[key], {
                  step: stepKey,
                  value: value,
                }].slice(-100);
              }
            }
          }
        });

        return updated;
      });
    }
  }, [progress, experimentType]);

  // Update stable outcomes - only increase, never decrease (prevents flickering from out-of-order WS messages)
  useEffect(() => {
    if (experimentType === 'Evaluation' && progress?.metrics) {
      const newSuccesses = progress.metrics.total_successes ?? 0;
      const newFailures = progress.metrics.total_failures ?? 0;
      const newTimeouts = progress.metrics.total_timeouts ?? 0;

      setStableOutcomes(prev => ({
        successes: Math.max(prev.successes, newSuccesses),
        failures: Math.max(prev.failures, newFailures),
        timeouts: Math.max(prev.timeouts, newTimeouts),
      }));
    }
  }, [progress?.metrics?.total_successes, progress?.metrics?.total_failures, progress?.metrics?.total_timeouts, experimentType]);

  // Reset history when modal closes
  useEffect(() => {
    if (!open) {
      setMetricsHistory({
        mean_reward: [],
        mean_ep_length: [],
      });
      setProgressHistory([]);
      setReplayTrajectory([]);
      setReplayFrame(null);
      setIsReplaying(false);
      setCurrentReplayIndex(0);
      setStableOutcomes({ successes: 0, failures: 0, timeouts: 0 });
    }
  }, [open]);

  // Replay playback effect
  useEffect(() => {
    if (!isReplaying || replayTrajectory.length === 0) return;

    const frameDelay = (1000 / 60) / replaySpeed; // 60 FPS adjusted by speed

    const interval = setInterval(() => {
      setCurrentReplayIndex(prev => {
        const next = prev + 1;
        if (next >= replayTrajectory.length) {
          setIsReplaying(false);
          return 0; // Reset to start
        }
        setReplayFrame(replayTrajectory[next] as SimulationFrame);
        return next;
      });
    }, frameDelay);

    return () => clearInterval(interval);
  }, [isReplaying, replayTrajectory, replaySpeed]);

  const startReplay = () => {
    if (replayTrajectory.length === 0) {
      alert('No trajectory data available for replay');
      return;
    }
    setCurrentReplayIndex(0);
    setReplayFrame(replayTrajectory[0] as SimulationFrame);
    setIsReplaying(true);
  };

  const stopReplay = () => {
    setIsReplaying(false);
    setCurrentReplayIndex(0);
    setReplayFrame(null);
  };

  const pauseReplay = () => {
    setIsReplaying(false);
  };

  const resumeReplay = () => {
    if (currentReplayIndex < replayTrajectory.length) {
      setIsReplaying(true);
    }
  };

  const handleRegenerate = async () => {
    if (!experimentId) return;
    try {
      await api.regenerateSimulation(experimentId);
      // Close modal and signal it should reopen after experiments reload
      onClose(true);
    } catch (err) {
      alert('Failed to regenerate simulation: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm"
    >
      <div
        className="w-full max-w-6xl max-h-[90vh] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <Card>
          <CardHeader
            title={
              <div className="flex items-center gap-3">
                <Activity className="h-5 w-5 text-notion-blue" />
                <span>{experimentName} - {experimentType} Info</span>
              </div>
            }
            actions={
              <button
                type="button"
                onClick={() => onClose()}
                className="rounded-xl p-2 text-notion-text-secondary transition hover:bg-notion-hover hover:text-notion-text"
              >
                <X className="h-5 w-5" />
              </button>
            }
          />

          <div className="overflow-y-auto max-h-[calc(90vh-120px)] p-6 space-y-6">
            {/* Loading state - show spinner until metrics are loaded */}
            {isLoadingHistory ? (
              <div className="flex flex-col items-center justify-center py-20 space-y-4">
                <div className="relative">
                  <div className="w-16 h-16 border-4 border-slate-700 border-t-cyan-500 rounded-full animate-spin"></div>
                  <Activity className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 h-6 w-6 text-notion-blue" />
                </div>
                <div className="text-notion-text-secondary text-lg font-medium">Loading metrics...</div>
                <div className="text-notion-text-tertiary text-sm">Please wait while we fetch experiment data</div>
              </div>
            ) : (
              <>
                {/* Progress Stats */}
                {(experimentType === 'Training' || experimentType === 'Fine-Tuning' || experimentType === 'Evaluation') && (progress || metricsHistory.mean_reward.length > 0) && (
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 text-notion-blue">
                      <BarChart3 className="h-5 w-5" />
                      <h3 className="font-semibold">{experimentType === 'Evaluation' ? 'Evaluation Progress' : 'Training Progress'}</h3>
                    </div>

                    {(() => {
                      // Get data from either live progress or historical metrics
                      // For evaluation mode, use episode-based progress
                      const isEvaluation = experimentType === 'Evaluation';
                      const totalEpisodes = (progress as any)?.total_episodes || experiment?.evaluation_episodes || 0;
                      // Fix: For completed evaluations, show total episodes as current
                      const currentEpisode = (experimentStatus === 'Completed' && isEvaluation)
                        ? totalEpisodes
                        : ((progress as any)?.episode || 0);
                      const currentStep = progress?.step || (metricsHistory.mean_reward.length > 0
                        ? metricsHistory.mean_reward[metricsHistory.mean_reward.length - 1].step
                        : 0);
                      const totalStepsValue = progress?.total_steps || totalSteps || 0;
                      const status = progress?.status || experimentStatus || 'Unknown';

                      // Get latest metrics from all available metric types
                      const latestMetrics = progress?.metrics || (() => {
                        if (Object.keys(metricsHistory).length === 0) return null;

                        const metrics: any = {};
                        Object.keys(metricsHistory).forEach(key => {
                          if (metricsHistory[key].length > 0) {
                            metrics[key] = metricsHistory[key][metricsHistory[key].length - 1].value;
                          }
                        });
                        return Object.keys(metrics).length > 0 ? metrics : null;
                      })();

                      // Format metric labels properly
                      const formatMetricLabel = (key: string): string => {
                        const labelMap: { [key: string]: string } = {
                          'reward_raw_mean': 'Mean Reward (Raw)',
                          'reward_shaped_mean': 'Mean Reward (Shaped)',
                          'success_rate': 'Success Rate',
                          'crash_rate': 'Crash Rate',
                          'timeout_rate': 'Timeout Rate',
                          'mean_ep_length': 'Mean Episode Length',
                          'cost_mean': 'Mean Episode Cost',
                          'violation_rate': 'Violation Rate',
                          'near_miss_mean': 'Near-Misses (avg)',
                          'danger_time_mean': 'Danger Time (avg)',
                          'lambda': 'Lambda (λ)',
                          'epsilon': 'Epsilon (ε)',
                          'reward_std': 'Reward Std Dev',
                          'cost_std': 'Cost Std Dev',
                        };
                        return labelMap[key] || key.split('_').map(word =>
                          word.charAt(0).toUpperCase() + word.slice(1)
                        ).join(' ');
                      };

                      return (
                        <>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4">
                              <div className="text-sm text-notion-text-secondary">{isEvaluation ? 'Current Episode' : 'Current Step'}</div>
                              <div className="text-2xl font-bold text-notion-blue">
                                {isEvaluation ? currentEpisode.toLocaleString() : currentStep.toLocaleString()}
                              </div>
                            </div>

                            {/* Modified: Show breakdown for Evaluation */}
                            {isEvaluation ? (
                              <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4">
                                <div className="text-sm text-notion-text-secondary">Outcomes</div>
                                <div className="flex items-baseline gap-2">
                                  <span className="text-emerald-400 font-bold text-xl" title="Successes">
                                    {stableOutcomes.successes}
                                  </span>
                                  <span className="text-notion-text-tertiary">/</span>
                                  <span className="text-rose-400 font-bold text-xl" title="Failures (Crashes)">
                                    {stableOutcomes.failures}
                                  </span>
                                  <span className="text-notion-text-tertiary">/</span>
                                  <span className="text-yellow-400 font-bold text-xl" title="Timeouts">
                                    {stableOutcomes.timeouts}
                                  </span>
                                </div>
                                <div className="text-[10px] text-notion-text-tertiary mt-1 flex justify-between">
                                  <span>Success</span>
                                  <span>Crash</span>
                                  <span>Timeout</span>
                                </div>
                              </div>
                            ) : (
                              <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4">
                                <div className="text-sm text-notion-text-secondary">Total Steps</div>
                                <div className="text-2xl font-bold text-notion-text">
                                  {totalStepsValue.toLocaleString()}
                                </div>
                              </div>
                            )}

                            <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4">
                              <div className="text-sm text-notion-text-secondary">Progress</div>
                              {isEvaluation ? (
                                <div className="flex flex-col">
                                  <div className="text-2xl font-bold text-emerald-400">
                                    {totalEpisodes > 0 ? `${((currentEpisode / totalEpisodes) * 100).toFixed(1)}%` : '0%'}
                                  </div>
                                  <div className="text-xs text-notion-text-tertiary mt-1">
                                    {currentEpisode} / {totalEpisodes}
                                  </div>
                                </div>
                              ) : (
                                <div className="text-2xl font-bold text-emerald-400">
                                  {totalStepsValue > 0 ? `${((currentStep / totalStepsValue) * 100).toFixed(1)}%` : '0%'}
                                </div>
                              )}
                            </div>

                            <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4">
                              <div className="text-sm text-notion-text-secondary">Status</div>
                              <div className={`text-xl font-bold ${status === 'Completed' ? 'text-emerald-400' :
                                status === 'Cancelled' ? 'text-orange-400' :
                                  status === 'In Progress' ? 'text-notion-blue' : 'text-notion-text-secondary'
                                }`}>
                                {status}
                              </div>
                            </div>
                          </div>
                        </>
                      );
                    })()}

                    {/* Comprehensive Metrics Charts */}
                    {progressHistory.length > 0 && (() => {
                      const safetyEnabled = experiment?.safety_constraint?.enabled || false;

                      console.log('[TrainingInfoModal] Safety Debug:', {
                        safetyConstraintEnabled: experiment?.safety_constraint?.enabled,
                        progressHistoryLength: progressHistory.length,
                        latestMetrics: progressHistory[progressHistory.length - 1]?.metrics,
                        safetyEnabled
                      });

                      return (
                        <div className="mt-6">
                          <MetricsCharts
                            progressHistory={progressHistory}
                            safetyEnabled={safetyEnabled}
                            useEpisodes={experiment?.type === 'Evaluation'}
                          />
                        </div>
                      );
                    })()}
                  </div>
                )}

                {/* Live Visualization - for Training/Evaluation in progress */}
                {(experimentType === 'Training' || experimentType === 'Fine-Tuning' || experimentType === 'Evaluation') && experimentStatus === 'In Progress' && frame && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-emerald-400">
                        <Activity className="h-5 w-5" />
                        <h3 className="font-semibold">{experimentType === 'Evaluation' ? 'Live Evaluation Visualization' : 'Live Training Visualization'}</h3>
                      </div>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={showVisualization}
                          onChange={(e) => setShowVisualization(e.target.checked)}
                          className="w-4 h-4 rounded border-slate-700 bg-notion-hover text-cyan-500 focus:ring-2 focus:ring-cyan-500"
                        />
                        <span className="text-sm text-notion-text-secondary">Show Visualization</span>
                      </label>
                    </div>

                    {showVisualization && (
                      <>
                        <SimulationViewer
                          frame={frame}
                          width={800}
                          height={500}
                          corridorLength={experiment?.environment?.length}
                          corridorWidth={experiment?.environment?.width}
                          corridorHeight={experiment?.environment?.height}
                          agentDiameter={experiment?.agent?.agent_diameter}
                          agentMaxSpeed={experiment?.agent?.agent_max_speed}
                        />

                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                          <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4">
                            <div className="text-sm text-notion-text-secondary">Current Frame</div>
                            <div className="text-2xl font-bold text-notion-blue">
                              {frame.frame || 0}
                            </div>
                          </div>

                          <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4">
                            <div className="text-sm text-notion-text-secondary">Episode</div>
                            <div className="text-2xl font-bold text-violet-400">
                              {frame.episode || 0}
                            </div>
                          </div>

                          <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4">
                            <div className="text-sm text-notion-text-secondary">Reward</div>
                            <div className={`text-2xl font-bold ${frame.reward > 0 ? 'text-emerald-400' : 'text-rose-400'
                              }`}>
                              {frame.reward?.toFixed(2) || '0.00'}
                            </div>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                )}

                {/* Trajectory Replay Viewer - for both Training and Simulation */}
                {replayTrajectory.length > 0 && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-notion-blue">
                        <Activity className="h-5 w-5" />
                        <h3 className="font-semibold">
                          {isReplaying ? 'Replay' : replayFrame ? 'Replay (Paused)' : (experimentType === 'Training' || experimentType === 'Fine-Tuning') ? 'Training Replay' : experimentType === 'Evaluation' ? 'Evaluation Replay' : 'Simulation Replay'}
                        </h3>
                      </div>

                      {/* Replay Controls */}
                      {(experimentStatus === 'Completed' || experimentStatus === 'Cancelled' || experimentStatus === 'Paused') && (
                        <div className="flex items-center gap-2">
                          {!isReplaying && !replayFrame && (
                            <>
                              {experimentType === 'Simulation' && (
                                <button
                                  onClick={handleRegenerate}
                                  className="rounded-lg border border-purple-200 bg-purple-50 px-3 py-1.5 text-sm text-purple-600 hover:bg-purple-900/40"
                                  title="Regenerate environment with new random configuration"
                                >
                                  🔄 Regenerate
                                </button>
                              )}
                              {(experimentStatus === 'Completed' || experimentStatus === 'Cancelled') && (
                                <button
                                  onClick={startReplay}
                                  className="rounded-lg border border-cyan-200 bg-cyan-50 px-3 py-1.5 text-sm text-cyan-600 hover:bg-cyan-900/40"
                                >
                                  ▶ Replay
                                </button>
                              )}
                            </>
                          )}
                          {isReplaying && (
                            <button
                              onClick={pauseReplay}
                              className="rounded-lg border border-yellow-200 bg-yellow-50 px-3 py-1.5 text-sm text-yellow-600 hover:bg-yellow-900/40"
                            >
                              ⏸ Pause
                            </button>
                          )}
                          {!isReplaying && replayFrame && (
                            <button
                              onClick={resumeReplay}
                              className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-sm text-emerald-600 hover:bg-emerald-900/40"
                            >
                              ▶ Resume
                            </button>
                          )}
                          {replayFrame && (
                            <button
                              onClick={stopReplay}
                              className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-1.5 text-sm text-rose-600 hover:bg-rose-900/40"
                            >
                              ⏹ Stop
                            </button>
                          )}
                          <select
                            value={replaySpeed}
                            onChange={(e) => setReplaySpeed(parseFloat(e.target.value))}
                            className="rounded-lg border border-notion-border bg-white px-2 py-1.5 text-sm text-notion-text"
                          >
                            <option value="0.25">0.25x</option>
                            <option value="0.5">0.5x</option>
                            <option value="1">1x</option>
                            <option value="2">2x</option>
                            <option value="4">4x</option>
                          </select>
                        </div>
                      )}
                    </div>

                    {/* Simulation Viewer - show replay frame if available, otherwise live frame */}
                    <SimulationViewer
                      frame={replayFrame || frame || null}
                      width={800}
                      height={500}
                      corridorLength={experiment?.environment?.length}
                      corridorWidth={experiment?.environment?.width}
                      corridorHeight={experiment?.environment?.height}
                      agentDiameter={experiment?.agent?.agent_diameter}
                      agentMaxSpeed={experiment?.agent?.agent_max_speed}
                    />

                    {/* Message when no frame data */}
                    {!(replayFrame || frame) && (experimentStatus === 'Completed' || experimentStatus === 'Cancelled') && (
                      <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4 text-center">
                        <div className="text-notion-text-secondary text-sm">
                          Click "▶ Replay" to start playback
                        </div>
                      </div>
                    )}

                    {/* Message when no frame data */}
                    {!(replayFrame || frame) && experimentStatus === 'Completed' && replayTrajectory.length === 0 && (
                      <div className="rounded-xl border border-notion-border bg-notion-light-gray p-8 text-center">
                        <div className="text-notion-text-secondary">
                          No simulation data available for replay.
                          <br />
                          <span className="text-sm text-notion-text-tertiary">This simulation was completed before trajectory recording was implemented.</span>
                        </div>
                      </div>
                    )}

                    {/* Frame stats */}
                    {(() => {
                      const displayFrame = replayFrame || frame;
                      if (!displayFrame) return null;

                      return (
                        <>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4">
                              <div className="text-sm text-notion-text-secondary">Frame</div>
                              <div className="text-2xl font-bold text-notion-blue">
                                {displayFrame.frame || 0}
                                {replayTrajectory.length > 0 && (
                                  <span className="text-xs text-notion-text-tertiary ml-1">/ {replayTrajectory.length}</span>
                                )}
                              </div>
                            </div>

                            <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4">
                              <div className="text-sm text-notion-text-secondary">Reward</div>
                              <div className={`text-2xl font-bold ${displayFrame.reward > 0 ? 'text-emerald-400' : 'text-rose-400'
                                }`}>
                                {displayFrame.reward?.toFixed(2) || '0.00'}
                              </div>
                            </div>

                            <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4">
                              <div className="text-sm text-notion-text-secondary">Status</div>
                              <div className={`text-xl font-bold ${displayFrame.success ? 'text-emerald-400' :
                                displayFrame.crashed ? 'text-rose-400' :
                                  displayFrame.done ? 'text-yellow-400' : 'text-blue-400'
                                }`}>
                                {displayFrame.success ? 'Success' :
                                  displayFrame.crashed ? 'Crashed' :
                                    displayFrame.done ? 'Done' : 'Running'}
                              </div>
                            </div>

                            <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4">
                              <div className="text-sm text-notion-text-secondary">Distance</div>
                              <div className="text-xl font-bold text-purple-400">
                                {displayFrame.agent_position && displayFrame.target_position ?
                                  Math.sqrt(
                                    Math.pow(displayFrame.agent_position.x - displayFrame.target_position.x, 2) +
                                    Math.pow(displayFrame.agent_position.y - displayFrame.target_position.y, 2) +
                                    Math.pow(displayFrame.agent_position.z - displayFrame.target_position.z, 2)
                                  ).toFixed(1) : 'N/A'}
                              </div>
                            </div>
                          </div>

                          {/* Agent Position & Velocity */}
                          {displayFrame.agent_position && (
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4">
                                <div className="text-sm text-notion-text-secondary mb-2">Position</div>
                                <div className="grid grid-cols-3 gap-3 text-sm">
                                  <div>
                                    <span className="text-notion-text-tertiary">X:</span>
                                    <span className="ml-2 text-notion-blue font-mono">{displayFrame.agent_position.x.toFixed(2)}</span>
                                  </div>
                                  <div>
                                    <span className="text-notion-text-tertiary">Y:</span>
                                    <span className="ml-2 text-notion-blue font-mono">{displayFrame.agent_position.y.toFixed(2)}</span>
                                  </div>
                                  <div>
                                    <span className="text-notion-text-tertiary">Z:</span>
                                    <span className="ml-2 text-notion-blue font-mono">{displayFrame.agent_position.z.toFixed(2)}</span>
                                  </div>
                                </div>
                              </div>

                              {displayFrame.agent_velocity && (
                                <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4">
                                  <div className="text-sm text-notion-text-secondary mb-2">Velocity</div>
                                  <div className="grid grid-cols-3 gap-3 text-sm">
                                    <div>
                                      <span className="text-notion-text-tertiary">X:</span>
                                      <span className="ml-2 text-yellow-400 font-mono">{displayFrame.agent_velocity.x.toFixed(2)}</span>
                                    </div>
                                    <div>
                                      <span className="text-notion-text-tertiary">Y:</span>
                                      <span className="ml-2 text-yellow-400 font-mono">{displayFrame.agent_velocity.y.toFixed(2)}</span>
                                    </div>
                                    <div>
                                      <span className="text-notion-text-tertiary">Z:</span>
                                      <span className="ml-2 text-yellow-400 font-mono">{displayFrame.agent_velocity.z.toFixed(2)}</span>
                                    </div>
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                        </>
                      );
                    })()}
                  </div>
                )}

                {/* No trajectory available message - for completed experiments without trajectory */}
                {replayTrajectory.length === 0 && (experimentStatus === 'Completed' || experimentStatus === 'Cancelled') && !isLoadingHistory && (
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 text-notion-text-tertiary">
                      <Activity className="h-5 w-5" />
                      <h3 className="font-semibold">Trajectory Replay</h3>
                    </div>

                    <div className="rounded-xl border border-notion-border bg-notion-light-gray p-8 text-center">
                      <div className="text-notion-text-secondary">
                        <div className="mb-2">No trajectory data available for this experiment.</div>
                        <div className="text-sm text-notion-text-tertiary">
                          This experiment was completed before trajectory recording was implemented.
                          <br />
                          Run a new {experimentType.toLowerCase()} experiment to see the replay feature.
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
