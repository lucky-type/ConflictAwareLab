import { useEffect, useMemo, useState, useCallback, useRef } from 'react';
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Activity, CheckCircle2, Download, GraduationCap, Joystick, Loader2, Lock, PauseCircle, Play, Plus, Trash2, X, XCircle, AlertCircle, Info } from 'lucide-react';
import { Badge } from '../components/common/Badge';
import { Card, CardHeader } from '../components/common/Card';
import NewExperimentModal from '../components/modals/NewExperimentModal';
import TrainingInfoModal from '../components/modals/TrainingInfoModal';
import { api, ExperimentWebSocket } from '../services/api';
import type { Experiment, ExperimentProgress, SimulationFrame } from '../services/api';
import { useExperimentWebSocket } from '../hooks/useExperimentWebSocket';
import { useTopBar } from '../context/TopBarContext';
import { useDataCache } from '../context/DataCacheContext';

type ExperimentType = 'Training' | 'Simulation' | 'Fine-Tuning' | 'Evaluation';

const FILTER_TABS: ReadonlyArray<'all' | ExperimentType> = ['all', 'Training', 'Fine-Tuning', 'Simulation', 'Evaluation'];

const typeVisuals: Record<
  ExperimentType,
  {
    label: string;
    icon: typeof GraduationCap;
    tone: string;
  }
> = {
  Training: {
    label: 'Training',
    icon: GraduationCap,
    tone: 'from-cyan-500/40 via-cyan-400/40 to-blue-500/40 text-cyan-100',
  },
  'Fine-Tuning': {
    label: 'Fine-Tuning',
    icon: GraduationCap,
    tone: 'from-orange-500/40 via-amber-400/40 to-yellow-500/40 text-orange-100',
  },
  Simulation: {
    label: 'Simulation',
    icon: Joystick,
    tone: 'from-violet-500/40 via-fuchsia-500/40 to-rose-500/40 text-violet-100',
  },
  Evaluation: {
    label: 'Evaluation',
    icon: Activity,
    tone: 'from-green-500/40 via-emerald-400/40 to-teal-500/40 text-green-100',
  },
};

const comparisonPalette = ['#22d3ee', '#f97316', '#a855f7', '#34d399'];

const statusVisuals: Record<string, { icon: typeof Loader2; accent: string; pulse?: boolean }> = {
  'In Progress': { icon: Loader2, accent: 'text-emerald-600', pulse: true },
  Completed: { icon: CheckCircle2, accent: 'text-cyan-600' },
  Paused: { icon: PauseCircle, accent: 'text-sky-600' },
  Cancelled: { icon: XCircle, accent: 'text-rose-600' },
};

const statusBadgeTone: Record<string, string> = {
  'In Progress': 'bg-emerald-100 text-emerald-700 border-emerald-200',
  Completed: 'bg-cyan-100 text-cyan-700 border-cyan-200',
  Paused: 'bg-sky-100 text-sky-700 border-sky-200',
  Cancelled: 'bg-rose-100 text-rose-700 border-rose-200',
};

export default function Experiments() {
  const { setActions } = useTopBar();
  const {
    completedMetrics,
    addCompletedExperimentMetrics,
    refreshMetrics,
    experiments,
    agents,
    environments,
    rewardFunctions,
    refreshStaticData,
    isLoadingStaticData,
  } = useDataCache();
  const [open, setOpen] = useState(false);
  const [infoModalOpen, setInfoModalOpen] = useState(false);
  const [filter, setFilter] = useState<'all' | ExperimentType>('all');
  const [selectedExperimentId, setSelectedExperimentId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [liveProgress, setLiveProgress] = useState<Record<number, ExperimentProgress>>({});
  const [liveFrame, setLiveFrame] = useState<Record<number, SimulationFrame>>({});
  const [reopenModalAfterLoad, setReopenModalAfterLoad] = useState(false);
  const [jobCapacity, setJobCapacity] = useState<{ running_jobs: number; max_concurrent_jobs: number; available_slots: number; can_start_new: boolean } | null>(null);

  useEffect(() => {
    setActions(
      <button
        onClick={() => setOpen(true)}
        className="flex items-center gap-2 rounded-md bg-notion-blue px-4 py-2 text-sm font-medium text-white hover:opacity-90 transition-opacity"
      >
        <Plus size={16} /> Create
      </button>
    );
    return () => setActions(null);
  }, []);

  // Refresh data after mutations (create/delete/start/pause etc.)
  const refreshData = useCallback(async () => {
    try {
      setError(null);
      await Promise.all([
        api.getJobCapacity().then(setJobCapacity),
        refreshStaticData(),
      ]);

      // Reopen modal if we just regenerated
      if (reopenModalAfterLoad) {
        setReopenModalAfterLoad(false);
        setInfoModalOpen(true);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load experiments');
      console.error('Failed to load experiments:', err);
    }
  }, [reopenModalAfterLoad, refreshStaticData]);

  // Load only job capacity on mount (static data already cached)
  useEffect(() => {
    const loadJobCapacity = async () => {
      try {
        const capacityData = await api.getJobCapacity();
        setJobCapacity(capacityData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load job capacity');
        console.error('Failed to load job capacity:', err);
      }
    };
    loadJobCapacity();
  }, []);

  // WebSocket connection for selected experiment
  const handleProgress = useCallback(async (data: ExperimentProgress) => {
    console.log('[Progress Update]', data);
    setLiveProgress((prev) => ({ ...prev, [data.experiment_id]: data }));

    // If experiment just completed/cancelled, add its metrics to cache and refresh experiments
    if ((data.status === 'Completed' || data.status === 'Cancelled') &&
      !completedMetrics[data.experiment_id]) {
      await addCompletedExperimentMetrics(data.experiment_id);
      await refreshStaticData(); // Refresh to get updated status
    }
  }, [completedMetrics, addCompletedExperimentMetrics, refreshStaticData]);

  const handleSimulationFrame = useCallback((data: SimulationFrame) => {
    setLiveFrame((prev) => ({ ...prev, [data.experiment_id]: data }));
  }, []);

  // Connect to all running experiments via WebSocket
  useEffect(() => {
    const runningExperiments = experiments.filter(exp => exp.status === 'In Progress');

    const connections = runningExperiments.map(exp => {
      const ws = new ExperimentWebSocket();
      ws.on('progress', handleProgress);
      ws.on('simulation_frame', handleSimulationFrame);
      ws.connect(exp.id);
      return ws;
    });

    return () => {
      connections.forEach(ws => {
        ws.off('progress', handleProgress);
        ws.off('simulation_frame', handleSimulationFrame);
        ws.disconnect();
      });
    };
  }, [experiments.map(e => e.id + e.status).join(','), handleProgress, handleSimulationFrame]);

  const filteredExperiments = useMemo(() => {
    if (filter === 'all') return experiments;
    return experiments.filter((exp) => exp.type === filter);
  }, [experiments, filter]);

  useEffect(() => {
    if (!filteredExperiments.length) {
      setSelectedExperimentId(null);
      return;
    }
    if (!selectedExperimentId || !filteredExperiments.some((exp) => exp.id === selectedExperimentId)) {
      setSelectedExperimentId(filteredExperiments[0]?.id ?? null);
    }
  }, [filteredExperiments.length, filter, selectedExperimentId]);

  const selectedExperiment = filteredExperiments.find((exp) => exp.id === selectedExperimentId) ?? null;
  const selectedProgress = selectedExperimentId ? liveProgress[selectedExperimentId] : null;
  const selectedFrame = selectedExperimentId ? liveFrame[selectedExperimentId] : null;

  const statusBuckets = useMemo(() => {
    return experiments.reduce(
      (acc, exp) => {
        acc.totalTraining += (exp.type === 'Training' || exp.type === 'Fine-Tuning') ? 1 : 0;
        acc.totalSimulation += exp.type === 'Simulation' ? 1 : 0;
        acc.inProgress += exp.status === 'In Progress' ? 1 : 0;
        acc.paused += exp.status === 'Paused' ? 1 : 0;
        acc.completed += exp.status === 'Completed' ? 1 : 0;
        acc.cancelled += exp.status === 'Cancelled' ? 1 : 0;
        return acc;
      },
      { totalTraining: 0, totalSimulation: 0, inProgress: 0, paused: 0, completed: 0, cancelled: 0 },
    );
  }, [experiments]);

  const handleDelete = async (id: number) => {
    if (!confirm('Are you sure you want to delete this experiment?')) return;

    try {
      await api.deleteExperiment(id);
      await refreshData();
      await refreshMetrics(); // Refresh cached metrics after deletion
      if (selectedExperimentId === id) {
        setSelectedExperimentId(null);
      }
    } catch (err) {
      alert('Failed to delete experiment: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  const handleStart = async (id: number) => {
    // Check job capacity before starting
    if (jobCapacity && !jobCapacity.can_start_new) {
      alert(`Cannot start new experiment: Maximum concurrent jobs (${jobCapacity.max_concurrent_jobs}) reached. Please wait for a job to complete.`);
      return;
    }

    try {
      await api.startExperiment(id);
      await refreshData();
      // Automatically select and open the experiment when started
      setTimeout(() => {
        setSelectedExperimentId(id);
        setInfoModalOpen(true);
      }, 300);
    } catch (err) {
      alert('Failed to start experiment: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  const handlePause = async (id: number) => {
    try {
      await api.pauseExperiment(id);
      await refreshData();
    } catch (err) {
      alert('Failed to pause experiment: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  const handleResume = async (id: number) => {
    try {
      await api.resumeExperiment(id);
      await refreshData();
    } catch (err) {
      alert('Failed to resume experiment: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  const handleDownload = async (id: number) => {
    try {
      // Create a temporary anchor element to trigger download
      const response = await fetch(`http://localhost:8000/experiments/${id}/export`);
      if (!response.ok) {
        throw new Error('Failed to download report');
      }

      // Get filename from Content-Disposition header or use default
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = `experiment_${id}_report.xlsx`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      }

      // Download the file
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      alert('Failed to download report: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  const handleCancel = async (id: number) => {
    if (!confirm('Are you sure you want to cancel this experiment?')) return;

    try {
      await api.cancelExperiment(id);
      await refreshData();
    } catch (err) {
      alert('Failed to cancel experiment: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  if (isLoadingStaticData) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-notion-blue" />
      </div>
    );
  }

  if (error) {
    return (
      <Card className="p-6">
        <div className="flex items-center gap-3 text-notion-red">
          <AlertCircle size={20} />
          <p>{error}</p>
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="p-6">
        <div className="flex items-center gap-3 text-notion-red">
          <AlertCircle size={20} />
          <p>{error}</p>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <SummaryCard label="Training" value={`${statusBuckets.totalTraining}`} hint="records" icon={GraduationCap} tone="from-cyan-600/30 to-blue-600/30" />
        <SummaryCard label="Simulation" value={`${statusBuckets.totalSimulation}`} hint="sessions" icon={Joystick} tone="from-fuchsia-600/30 to-rose-600/30" />
        <SummaryCard label="Active" value={`${statusBuckets.inProgress}`} hint="in progress" icon={Loader2} tone="from-emerald-600/30 to-teal-600/30" animated />
        <SummaryCard label="Paused" value={`${statusBuckets.paused}`} hint="waiting action" icon={PauseCircle} tone="from-slate-700/60 to-slate-900/60" />
      </div>

      <Card className="space-y-4">
        <CardHeader
          title="Experiment List"
          actions={
            <div className="flex flex-wrap items-center gap-2">
              {jobCapacity && (
                <div className={`flex items-center gap-2 rounded-xl border px-3 py-1.5 text-xs font-semibold ${jobCapacity.available_slots === 0
                  ? 'border-rose-500/40 bg-rose-500/10 text-rose-600'
                  : jobCapacity.available_slots <= 2
                    ? 'border-amber-500/40 bg-amber-500/10 text-amber-600'
                    : 'border-emerald-500/40 bg-emerald-500/10 text-emerald-600'
                  }`}>
                  <Activity size={12} />
                  {jobCapacity.running_jobs}/{jobCapacity.max_concurrent_jobs} Jobs
                </div>
              )}
              {FILTER_TABS.map((tab) => (
                <button
                  key={tab}
                  onClick={() => setFilter(tab)}
                  className={`rounded-xl border px-3 py-1.5 text-xs font-semibold transition ${filter === tab
                    ? 'border-cyan-500 bg-cyan-100 text-cyan-700'
                    : 'border-notion-border text-notion-text-secondary hover:border-notion-border-dark hover:text-notion-text'
                    }`}
                >
                  {tab === 'all' ? 'All' : tab}
                </button>
              ))}
            </div>
          }
        />

        {filteredExperiments.length === 0 ? (
          <p className="rounded-xl border border-dashed border-notion-border bg-notion-light-gray px-4 py-6 text-center text-sm text-notion-text-secondary">
            No experiments in this category.
          </p>
        ) : (
          <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
            {filteredExperiments.map((exp) => (
              <ExperimentCard
                key={exp.id}
                experiment={exp}
                progress={liveProgress[exp.id] || completedMetrics[exp.id]}
                active={selectedExperiment?.id === exp.id}
                onOpen={() => {
                  setSelectedExperimentId(exp.id);
                  setInfoModalOpen(true);
                }}
                onDelete={() => handleDelete(exp.id)}
                onStart={() => handleStart(exp.id)}
                onPause={() => handlePause(exp.id)}
                onResume={() => handleResume(exp.id)}
                onCancel={() => handleCancel(exp.id)}
                onInfo={() => {
                  setSelectedExperimentId(exp.id);
                  setInfoModalOpen(true);
                }}
                onDownload={() => handleDownload(exp.id)}
                algorithms={agents}
                environments={environments}
              />
            ))}
          </div>
        )}
      </Card>

      <NewExperimentModal
        open={open}
        onClose={() => setOpen(false)}
        onSave={async (experimentId?: number) => {
          await refreshData();
          // Automatically select and open the newly created experiment
          if (experimentId) {
            // Use setTimeout to ensure the experiment list is updated and WebSocket connects
            setTimeout(() => {
              setSelectedExperimentId(experimentId);
              setInfoModalOpen(true);
            }, 300);
          }
        }}
      />

      <TrainingInfoModal
        open={infoModalOpen}
        onClose={(shouldReopen?: boolean) => {
          setInfoModalOpen(false);
          // Only set flag if regenerate was called (shouldReopen = true)
          if (shouldReopen === true) {
            setReopenModalAfterLoad(true);
          } else {
            // Clear flag if manually closing
            setReopenModalAfterLoad(false);
          }
          // Reload experiments when modal closes to get latest status
          refreshData();
        }}
        experimentName={selectedExperiment?.name || 'Experiment'}
        experimentType={selectedExperiment?.type || 'Training'}
        experimentId={selectedExperiment?.id}
        experimentStatus={selectedExperiment?.status}
        totalSteps={selectedExperiment?.total_steps}
        progress={selectedExperiment ? liveProgress[selectedExperiment.id] : undefined}
        frame={selectedExperiment ? liveFrame[selectedExperiment.id] : undefined}
        experiment={selectedExperiment || undefined}
      />
    </div>
  );
}

type CardProps = {
  label: string;
  value: string;
  hint?: string;
  icon: typeof Activity;
  tone: string;
  animated?: boolean;
};

function SummaryCard({ label, value, hint, icon: Icon, tone, animated }: CardProps) {
  return (
    <div className="rounded-lg border border-notion-border bg-notion-light-gray p-4">
      <div className={`mb-3 inline-flex rounded-lg bg-notion-blue/10 p-2 text-notion-blue ${animated ? 'animate-pulse' : ''}`}>
        <Icon size={18} />
      </div>
      <p className="text-xs uppercase tracking-wide text-notion-text-secondary">{label}</p>
      <div className="text-2xl font-semibold text-notion-text">{value}</div>
      {hint && <p className="text-xs text-notion-text-tertiary">{hint}</p>}
    </div>
  );
}

type ExperimentCardProps = {
  experiment: Experiment;
  progress?: ExperimentProgress;
  active: boolean;
  onOpen: () => void;
  onDelete: () => void;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
  onInfo: () => void;
  onDownload: () => void;
  algorithms: any[];
  environments: any[];
};

function ExperimentCard({ experiment, progress, active, onOpen, onDelete, onStart, onPause, onResume, onCancel, onInfo, onDownload, algorithms, environments }: ExperimentCardProps) {
  const statusInfo = statusVisuals[experiment.status] ?? { icon: Activity, accent: 'text-notion-text' };
  const StatusIcon = statusInfo.icon;
  const typeKey = experiment.type;
  const TypeIcon = typeVisuals[typeKey].icon;

  // Stable outcomes state to prevent flickering from out-of-order WS messages
  const [stableOutcomes, setStableOutcomes] = useState({ successes: 0, failures: 0, timeouts: 0 });

  // Update stable outcomes - only increase, never decrease
  useEffect(() => {
    if (progress?.metrics) {
      const newSuccesses = progress.metrics.total_successes ?? 0;
      const newFailures = progress.metrics.total_failures ?? 0;
      const newTimeouts = progress.metrics.total_timeouts ?? 0;

      setStableOutcomes(prev => ({
        successes: Math.max(prev.successes, newSuccesses),
        failures: Math.max(prev.failures, newFailures),
        timeouts: Math.max(prev.timeouts, newTimeouts),
      }));
    }
  }, [progress?.metrics?.total_successes, progress?.metrics?.total_failures, progress?.metrics?.total_timeouts, experiment.type]);

  // Reset stable outcomes when experiment changes or completes
  useEffect(() => {
    if (experiment.status !== 'In Progress') {
      // When completed, use final values from progress if available
      if (progress?.metrics) {
        setStableOutcomes({
          successes: progress.metrics.total_successes ?? 0,
          failures: progress.metrics.total_failures ?? 0,
          timeouts: progress.metrics.total_timeouts ?? 0,
        });
      }
    } else {
      // Reset when starting a new evaluation
      setStableOutcomes({ successes: 0, failures: 0, timeouts: 0 });
    }
  }, [experiment.id, experiment.status]);

  // Helper functions to get names
  const getAlgorithmName = (id: number | null | undefined) => {
    if (!id) return 'N/A';
    return algorithms.find(a => a.id === id)?.name || `Algorithm ${id}`;
  };
  const getEnvironmentName = (id: number | null | undefined) => {
    if (!id) return 'N/A';
    return environments.find(e => e.id === id)?.name || `Environment ${id}`;
  };

  const currentStep = progress?.step || experiment.current_step || 0;
  const progressPercent = experiment.total_steps
    ? Math.round((currentStep / experiment.total_steps) * 100)
    : 0;

  // For evaluation mode, show episode progress
  // Fix: For completed evaluations, show the total episodes as current (all done)
  const totalEpisodes = experiment.evaluation_episodes || 0;
  const currentEpisode = experiment.status === 'Completed' && experiment.type === 'Evaluation'
    ? totalEpisodes  // All episodes completed
    : (progress?.episode || 0);
  const episodeProgressPercent = totalEpisodes
    ? Math.round((currentEpisode / totalEpisodes) * 100)
    : 0;

  return (
    <div
      className={`flex h-full flex-col rounded-2xl border-2 p-4 transition ${active ? 'border-cyan-500/60 bg-notion-light-gray shadow-xl shadow-cyan-500/10' : 'border-slate-900 bg-notion-light-gray'
        }`}
      onClick={onOpen}
    >
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <span className="inline-flex items-center gap-1 rounded-full border border-notion-border px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-notion-text">
            <TypeIcon size={12} /> {experiment.type}
          </span>
          <h4 className="text-base font-semibold text-notion-text">{experiment.name}</h4>
          <p className="text-xs text-notion-text-secondary">
            {getEnvironmentName(experiment.env_id)} • {getAlgorithmName(experiment.agent_id)}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {progress?.metrics?.success_rate != null && (
            <div className="flex items-center gap-1 rounded-full bg-emerald-500/20 px-2 py-0.5 text-xs font-semibold text-emerald-600 border border-emerald-500/30">
              <CheckCircle2 size={10} />
              {progress.metrics.success_rate.toFixed(1)}%
            </div>
          )}
          {progress?.metrics?.crash_rate != null && (
            <div className="flex items-center gap-1 rounded-full bg-rose-500/20 px-2 py-0.5 text-xs font-semibold text-rose-600 border border-rose-500/30">
              <XCircle size={10} />
              {progress.metrics.crash_rate.toFixed(1)}%
            </div>
          )}
          <Badge label={experiment.status} tone={statusBadgeTone[experiment.status] || 'bg-slate-500/20 text-notion-text'} />
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-notion-text-secondary">
        <div className={`flex items-center gap-1 text-sm ${statusInfo.accent} ${statusInfo.pulse ? 'animate-pulse' : ''}`}>
          <StatusIcon size={16} />
          {experiment.status}
        </div>
        {(experiment.type === 'Training' || experiment.type === 'Fine-Tuning') && experiment.total_steps && (
          <>
            <span>•</span>
            <span>{currentStep.toLocaleString()} / {experiment.total_steps.toLocaleString()} steps</span>
          </>
        )}
        {experiment.type === 'Evaluation' && experiment.evaluation_episodes && (
          <>
            <span>•</span>
            <span>{currentEpisode.toLocaleString()} / {experiment.evaluation_episodes.toLocaleString()} episodes</span>
            {(stableOutcomes.successes > 0 || stableOutcomes.failures > 0 || stableOutcomes.timeouts > 0) && (
              <>
                <span className="text-slate-600 mx-1">|</span>
                <span className="flex items-center gap-1 text-emerald-400 font-medium">
                  <CheckCircle2 size={12} />
                  {stableOutcomes.successes}
                </span>
                {stableOutcomes.failures > 0 && (
                  <span className="flex items-center gap-1 text-rose-400 font-medium">
                    <XCircle size={12} />
                    {stableOutcomes.failures}
                  </span>
                )}
                {stableOutcomes.timeouts > 0 && (
                  <span className="flex items-center gap-1 text-yellow-400 font-medium">
                    <AlertCircle size={12} />
                    {stableOutcomes.timeouts}
                  </span>
                )}
              </>
            )}
          </>
        )}
      </div>

      {(experiment.type === 'Training' || experiment.type === 'Fine-Tuning') && experiment.total_steps && (
        <div className="mt-4 flex flex-col gap-2 text-sm text-notion-text">
          <div className="flex justify-between text-xs text-notion-text-secondary">
            <span>Progress</span>
            <span>{progressPercent}%</span>
          </div>
          <div className="h-1.5 rounded-full bg-notion-hover">
            <div className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-cyan-500" style={{ width: `${progressPercent}%` }} />
          </div>
          {(stableOutcomes.successes > 0 || stableOutcomes.failures > 0 || stableOutcomes.timeouts > 0) && (
            <div className="mt-2 flex items-center gap-3 text-xs">
              <span className="text-notion-text-tertiary">Episodes:</span>
              {stableOutcomes.successes > 0 && (
                <span className="flex items-center gap-1 text-emerald-400 font-medium">
                  <CheckCircle2 size={12} />
                  {stableOutcomes.successes}
                </span>
              )}
              {stableOutcomes.failures > 0 && (
                <span className="flex items-center gap-1 text-rose-400 font-medium">
                  <XCircle size={12} />
                  {stableOutcomes.failures}
                </span>
              )}
              {stableOutcomes.timeouts > 0 && (
                <span className="flex items-center gap-1 text-yellow-400 font-medium">
                  <AlertCircle size={12} />
                  {stableOutcomes.timeouts}
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {experiment.type === 'Evaluation' && experiment.evaluation_episodes && (
        <div className="mt-4 flex flex-col gap-2 text-sm text-notion-text">
          <div className="flex justify-between text-xs text-notion-text-secondary">
            <span>Progress</span>
            <span>{episodeProgressPercent}%</span>
          </div>
          <div className="h-1.5 rounded-full bg-notion-hover">
            <div className="h-full rounded-full bg-gradient-to-r from-green-400 to-teal-500" style={{ width: `${episodeProgressPercent}%` }} />
          </div>
        </div>
      )}

      <div className="mt-auto flex gap-2 pt-4" onClick={(e) => e.stopPropagation()}>
        {/* Download button - always visible in bottom-left */}
        <button
          onClick={onDownload}
          className="flex items-center justify-center rounded-xl border border-notion-border px-3 py-2 text-sm text-notion-text hover:border-cyan-500/60 hover:text-notion-blue"
          title="Download Excel Report"
        >
          <Download size={14} />
        </button>
        {(experiment.status === 'In Progress' || experiment.status === 'Paused') && (
          <button
            onClick={onInfo}
            className="flex items-center justify-center rounded-xl border border-cyan-200 px-3 py-2 text-sm text-cyan-600 hover:border-cyan-500/60"
            title="View training info"
          >
            <Info size={14} />
          </button>
        )}
        {experiment.status === 'Paused' && (
          <button
            onClick={onResume}
            className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-emerald-200 px-3 py-2 text-sm text-emerald-600 hover:border-emerald-500/60"
          >
            <Play size={14} /> Resume
          </button>
        )}
        {experiment.status === 'In Progress' && (
          <button
            onClick={onPause}
            className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-sky-200 px-3 py-2 text-sm text-sky-600 hover:border-sky-500/60"
          >
            <PauseCircle size={14} /> Pause
          </button>
        )}
        {experiment.status === 'Completed' && (
          <button
            onClick={onOpen}
            className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-notion-border px-3 py-2 text-sm text-notion-text hover:border-cyan-400/60"
          >
            <Activity size={14} /> View
          </button>
        )}
        {experiment.status === 'Cancelled' && (
          <button
            onClick={onOpen}
            className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-notion-border px-3 py-2 text-sm text-notion-text hover:border-cyan-400/60"
          >
            <Activity size={14} /> View
          </button>
        )}
        {(experiment.status === 'In Progress' || experiment.status === 'Paused') && (
          <button
            onClick={onCancel}
            className="flex items-center justify-center rounded-xl border border-rose-200 px-3 py-2 text-sm text-rose-600 hover:border-rose-500/60"
          >
            <XCircle size={14} />
          </button>
        )}
        {(experiment.status === 'Completed' || experiment.status === 'Cancelled') && (
          <button
            onClick={onDelete}
            className="flex items-center justify-center rounded-xl border border-rose-200 px-3 py-2 text-sm text-rose-600 hover:border-rose-500/60"
          >
            <Trash2 size={14} />
          </button>
        )}
      </div>
    </div>
  );
}

function InfoRow({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-notion-text-secondary">{label}</span>
      <span className={accent ?? 'text-notion-text'}>{value}</span>
    </div>
  );
}

type SimulationProps = {
  frame: SimulationFrame;
};

function SimulationViewport({ frame }: SimulationProps) {
  const { agent_position, obstacles, target_position, lidar_readings, reward, done } = frame;

  return (
    <div className="relative h-96 rounded-2xl border border-notion-border bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-4">
      <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet" className="h-full w-full">
        <defs>
          <linearGradient id="sim-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#22d3ee" />
            <stop offset="100%" stopColor="#a855f7" />
          </linearGradient>
        </defs>

        {/* Grid */}
        <g stroke="#1e293b" strokeWidth="0.2">
          {Array.from({ length: 10 }).map((_, idx) => (
            <line key={`row-${idx}`} x1="0" x2="100" y1={(idx + 1) * 10} y2={(idx + 1) * 10} opacity="0.3" />
          ))}
          {Array.from({ length: 10 }).map((_, idx) => (
            <line key={`col-${idx}`} y1="0" y2="100" x1={(idx + 1) * 10} x2={(idx + 1) * 10} opacity="0.3" />
          ))}
        </g>

        {/* Obstacles */}
        {obstacles?.map((obstacle, index) => (
          <circle
            key={`ob-${index}`}
            cx={obstacle.x}
            cy={100 - obstacle.y}
            r={obstacle.r || 3}
            fill="rgba(248,113,113,0.25)"
            stroke="#f87171"
            strokeWidth="0.5"
          />
        ))}

        {/* Target */}
        {target_position && (
          <circle cx={target_position.x} cy={100 - target_position.y} r="2.5" fill="#facc15" stroke="#fbbf24" strokeWidth="0.5" />
        )}

        {/* Agent */}
        {agent_position && (
          <circle cx={agent_position.x} cy={100 - agent_position.y} r="2" fill="#22d3ee" stroke="#06b6d4" strokeWidth="0.5" />
        )}

        {/* Lidar readings */}
        {lidar_readings && agent_position && lidar_readings.map((reading, idx) => {
          const angle = (idx / lidar_readings.length) * Math.PI * 2;
          const x2 = agent_position.x + Math.cos(angle) * reading * 10;
          const y2 = (100 - agent_position.y) - Math.sin(angle) * reading * 10;
          return (
            <line
              key={`lidar-${idx}`}
              x1={agent_position.x}
              y1={100 - agent_position.y}
              x2={x2}
              y2={y2}
              stroke="#a855f7"
              strokeWidth="0.2"
              opacity="0.3"
            />
          );
        })}
      </svg>

      <div className="absolute bottom-4 left-4 flex gap-3">
        <div className="rounded-xl border border-notion-border/60 bg-notion-light-gray px-3 py-1 text-xs text-notion-text">
          Frame: {frame.frame}
        </div>
        <div className={`rounded-xl border px-3 py-1 text-xs ${reward && reward > 0
          ? 'border-emerald-200/60 bg-emerald-900/40 text-emerald-600'
          : 'border-rose-200/60 bg-rose-900/40 text-rose-600'
          }`}>
          Reward: {reward?.toFixed(2) || '0.00'}
        </div>
        {done && (
          <div className="rounded-xl border border-cyan-200/60 bg-cyan-900/40 px-3 py-1 text-xs text-notion-blue">
            Episode Complete
          </div>
        )}
      </div>
    </div>
  );
}
