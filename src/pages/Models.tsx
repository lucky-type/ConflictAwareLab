import { useEffect, useState } from 'react';
import { Info, Layers3, Shield, Trash2, GitCompare, Search, ArrowUpDown } from 'lucide-react';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { api, type Experiment, type ModelSnapshot } from '../services/api';
import TrainingInfoModal from '../components/modals/TrainingInfoModal';
import CompareModelsModal from '../components/modals/CompareModelsModal';
import { useTopBar } from '../context/TopBarContext';
import { useDataCache } from '../context/DataCacheContext';

interface ExperimentWithSnapshots extends Experiment {
  snapshots: ModelSnapshot[];
}

export default function Models() {
  const {
    snapshotsByExperiment,
    isLoadingSnapshots,
    refreshSnapshots,
    experiments: cachedExperiments,
    environments,
    agents,
    rewardFunctions,
    isLoadingStaticData,
  } = useDataCache();
  const [experiments, setExperiments] = useState<ExperimentWithSnapshots[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedSnapshot, setSelectedSnapshot] = useState<{
    experiment: ExperimentWithSnapshots;
    snapshot: ModelSnapshot;
  } | null>(null);
  const [selectedSnapshotsForComparison, setSelectedSnapshotsForComparison] = useState<Set<number>>(new Set());
  const [showCompareModal, setShowCompareModal] = useState(false);


  // Filtering and Sorting state
  const [filterName, setFilterName] = useState('');
  const [sortBy, setSortBy] = useState<'date' | 'success' | 'failure' | 'reward'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  const { setActions } = useTopBar();

  useEffect(() => {
    if (!isLoadingSnapshots && !isLoadingStaticData) {
      loadModels();
    }
  }, [isLoadingSnapshots, isLoadingStaticData, cachedExperiments, snapshotsByExperiment]);

  useEffect(() => {
    if (selectedSnapshotsForComparison.size >= 2) {
      setActions(
        <div className="flex items-center gap-2">
          <button
            onClick={handleBulkDelete}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-rose-600 to-red-500 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-rose-500/20 hover:from-rose-500 hover:to-red-400 transition-all"
          >
            <Trash2 size={16} />
            Delete Selected ({selectedSnapshotsForComparison.size})
          </button>
          <button
            onClick={handleCompare}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-notion-blue to-notion-blue px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-notion-blue/20"
          >
            <GitCompare size={16} />
            Compare ({selectedSnapshotsForComparison.size})
          </button>
        </div>
      );
    } else {
      setActions(null);
    }
    return () => setActions(null);
  }, [selectedSnapshotsForComparison.size]);

  const loadModels = () => {
    try {
      setError(null);

      // Attach cached snapshots to experiments
      const experimentsWithSnapshots = cachedExperiments.map(exp => ({
        ...exp,
        snapshots: snapshotsByExperiment[exp.id] || []
      }));

      // Only show experiments that have snapshots
      setExperiments(experimentsWithSnapshots.filter(exp => exp.snapshots.length > 0));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models');
      console.error('Failed to load models:', err);
    }
  };

  const handleDeleteSnapshot = async (experimentId: number, snapshotId: number) => {
    if (!confirm('Delete this model snapshot? This action cannot be undone.')) return;

    try {
      await api.deleteSnapshot(snapshotId);
      // Refresh snapshots cache and reload models
      await refreshSnapshots();
      await loadModels();
    } catch (err) {
      alert(`Failed to delete snapshot: ${err instanceof Error ? err.message : 'Unknown error'}`);
      console.error('Failed to delete snapshot:', err);
    }
  };

  const handleBulkDelete = async () => {
    const count = selectedSnapshotsForComparison.size;
    if (!confirm(`Delete ${count} selected model snapshots? This action cannot be undone.`)) return;

    try {
      const snapshotIds = Array.from(selectedSnapshotsForComparison);
      const result = await api.deleteSnapshots(snapshotIds);

      if (result.failed_ids && result.failed_ids.length > 0) {
        alert(`Deleted ${result.deleted_ids.length} snapshots. ${result.failed_ids.length} failed.`);
      }

      // Clear selection, refresh cache, and reload
      setSelectedSnapshotsForComparison(new Set());
      await refreshSnapshots();
      await loadModels();
    } catch (err) {
      alert(`Failed to delete snapshots: ${err instanceof Error ? err.message : 'Unknown error'}`);
      console.error('Failed to bulk delete snapshots:', err);
    }
  };

  const handleViewSnapshot = (experiment: ExperimentWithSnapshots, snapshot: ModelSnapshot) => {
    setSelectedSnapshot({ experiment, snapshot });
  };

  const toggleSnapshotSelection = (snapshotId: number) => {
    setSelectedSnapshotsForComparison(prev => {
      const newSet = new Set(prev);
      if (newSet.has(snapshotId)) {
        newSet.delete(snapshotId);
      } else {
        newSet.add(snapshotId);
      }
      return newSet;
    });
  };

  const getSelectedSnapshots = (): ModelSnapshot[] => {
    const snapshots: ModelSnapshot[] = [];
    experiments.forEach(exp => {
      exp.snapshots.forEach(snapshot => {
        if (selectedSnapshotsForComparison.has(snapshot.id)) {
          snapshots.push(snapshot);
        }
      });
    });
    return snapshots;
  };

  const handleCompare = () => {
    setShowCompareModal(true);
  };

  const statusBadgeTone: Record<string, string> = {
    'Completed': 'bg-emerald-100 text-emerald-700 border-emerald-200',
    'In Progress': 'bg-cyan-100 text-cyan-700 border-cyan-200',
    'Paused': 'bg-orange-100 text-orange-700 border-orange-200',
    'Cancelled': 'bg-red-100 text-red-700 border-red-200',
  };

  // Filter and Sort Experiments
  const filteredExperiments = experiments
    .filter(exp => {
      const lowerFilter = filterName.toLowerCase();
      const matchesName = exp.name.toLowerCase().includes(lowerFilter);

      // Also match if any snapshot (name/iteration) matches
      const matchesSnapshot = exp.snapshots.some(s => {
        const sName = `${exp.name}_model+_${s.iteration}`;
        return sName.toLowerCase().includes(lowerFilter);
      });

      return matchesName || matchesSnapshot;
    })
    .sort((a, b) => {
      // Helper to get metric from latest snapshot (or any snapshot)
      // We use the LATEST snapshot (highest iteration) for sorting the experiment
      const getMetric = (exp: ExperimentWithSnapshots, metric: string): number => {
        if (!exp.snapshots || exp.snapshots.length === 0) return -1;
        // Sort snapshots by iteration desc to get latest
        const sortedSnaps = [...exp.snapshots].sort((x, y) => y.iteration - x.iteration);
        const latest = sortedSnaps[0];

        if (!latest.metrics_at_save) return -1;

        switch (metric) {
          case 'success':
            return latest.metrics_at_save.success_rate || 0;
          case 'failure':
            return latest.metrics_at_save.failure_rate || 0;
          case 'reward':
            return latest.metrics_at_save.reward || -1000;
          default: // date
            return new Date(latest.created_at).getTime();
        }
      };

      let valA, valB;
      if (sortBy === 'date') {
        valA = new Date(a.created_at).getTime();
        valB = new Date(b.created_at).getTime();
      } else {
        valA = getMetric(a, sortBy);
        valB = getMetric(b, sortBy);
      }

      if (sortOrder === 'asc') {
        return valA > valB ? 1 : -1;
      } else {
        return valA < valB ? 1 : -1;
      }
    });

  return (
    <div className="space-y-6">
      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Header */}
      <div>
        <p className="text-xs uppercase tracking-wide text-notion-blue">Models</p>
        <h1 className="text-2xl font-semibold text-notion-text">Model Registry</h1>
        <p className="text-sm text-notion-text-secondary">
          Browse and manage trained model snapshots. Each model represents at least one training run that saved a checkpoint.
        </p>
      </div>

      {/* Info Card */}
      <Card>
        <div className="flex items-start gap-3 rounded-lg border border-notion-border bg-notion-light-gray p-4">
          <div className="rounded-lg bg-blue-50 p-2 text-notion-blue">
            <Info size={20} />
          </div>
          <div className="text-sm text-notion-text-secondary">
            <p>
              Models are named using the format <span className="font-mono text-notion-blue">experiment_name_model</span>,
              with each snapshot suffixed by <span className="font-mono text-notion-blue">+_iteration</span> for easy identification.
            </p>
          </div>
        </div>
      </Card>

      {/* Controls: Filter and Sort */}
      < div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between" >
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-notion-text-secondary" />
          <input
            type="text"
            placeholder="Filter by experiment name..."
            value={filterName}
            onChange={(e) => setFilterName(e.target.value)}
            className="w-full rounded-lg border border-notion-border bg-white py-2 pl-9 pr-4 text-sm text-notion-text placeholder-notion-text-tertiary focus:border-notion-blue focus:outline-none focus:ring-1 focus:ring-notion-blue/30"
          />
        </div>

        <div className="flex items-center gap-3">
          <span className="text-sm text-notion-text-secondary">Sort by:</span>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="rounded-lg border border-notion-border bg-white px-3 py-1.5 text-sm text-notion-text focus:border-notion-blue focus:outline-none"
          >
            <option value="date">Date Created</option>
            <option value="success">Success Rate</option>
            <option value="failure">Failure Rate</option>
            <option value="reward">Reward Value</option>
          </select>

          <button
            onClick={() => setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc')}
            className="p-1.5 rounded-lg border border-notion-border bg-white text-notion-text-secondary hover:text-notion-blue hover:border-notion-blue/30 transition"
            title={sortOrder === 'asc' ? "Ascending" : "Descending"}
          >
            <ArrowUpDown size={16} />
          </button>
        </div>
      </div >

      {
        (isLoadingSnapshots || isLoadingStaticData) ? (
          <div className="text-center text-notion-text-secondary py-12" > Loading models...</div>
        ) : experiments.length === 0 ? (
          <div className="text-center text-notion-text-secondary py-12">
            No trained models yet. Start an experiment to create model snapshots.
          </div>
        ) : (
          <div className="space-y-5">
            {filteredExperiments.map((entry) => {
              const modelName = `${entry.name}_model`;

              // Helper functions to get names
              const getEnvironmentName = (id: number) => {
                return environments.find(e => e.id === id)?.name || `Environment ${id}`;
              };
              const getAlgorithmName = (id: number) => {
                return agents.find(a => a.id === id)?.name || `Algorithm ${id}`;
              };
              const getRewardFunctionName = (id: number) => {
                return rewardFunctions.find(r => r.id === id)?.name || `Reward Function ${id}`;
              };

              return (
                <Card key={entry.id} className="space-y-4">
                  <CardHeader
                    title={modelName}
                    subtitle={`Experiment #${entry.id} • ${entry.type}`}
                    actions={<Badge label={entry.status} tone={statusBadgeTone[entry.status] || 'bg-slate-500/20 text-notion-text'} />}
                  />

                  <div className="grid gap-3 text-xs text-notion-text-secondary md:grid-cols-4">
                    <div>
                      <p className="text-notion-text-tertiary">Environment</p>
                      <p className="text-notion-text">{getEnvironmentName(entry.env_id)}</p>
                    </div>
                    <div>
                      <p className="text-notion-text-tertiary">Algorithm</p>
                      <p className="text-notion-text">{getAlgorithmName(entry.agent_id)}</p>
                    </div>
                    <div>
                      <p className="text-notion-text-tertiary">Reward Function</p>
                      <p className="text-notion-text">{getRewardFunctionName(entry.reward_id)}</p>
                    </div>
                    <div>
                      <p className="text-notion-text-tertiary">Created</p>
                      <p className="text-notion-text">{new Date(entry.created_at).toLocaleDateString()}</p>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <p className="text-xs uppercase tracking-wide text-notion-text-secondary">
                      Snapshots ({entry.snapshots.length})
                    </p>

                    {/* Filter and Sort Snapshots for Display */}
                    {(() => {
                      const lowerFilter = filterName.toLowerCase();
                      const processedSnapshots = entry.snapshots
                        .filter(snapshot => {
                          const snapshotName = `${modelName}+_${snapshot.iteration}`;
                          return snapshotName.toLowerCase().includes(lowerFilter);
                        })
                        .sort((a, b) => {
                          const getSnapMetric = (s: ModelSnapshot, m: string): number => {
                            if (!s.metrics_at_save) return -1;
                            switch (m) {
                              case 'success': return s.metrics_at_save.success_rate || 0;
                              case 'failure': return s.metrics_at_save.failure_rate || 0;
                              case 'reward': return s.metrics_at_save.reward || -1000;
                              default: return new Date(s.created_at).getTime();
                            }
                          };

                          let valA, valB;
                          if (sortBy === 'date') {
                            valA = new Date(a.created_at).getTime();
                            valB = new Date(b.created_at).getTime();
                          } else {
                            valA = getSnapMetric(a, sortBy);
                            valB = getSnapMetric(b, sortBy);
                          }

                          // If values are equal (e.g. same reward), break tie with iteration (descending usually preferred)
                          if (valA === valB) {
                            return b.iteration - a.iteration;
                          }

                          if (sortOrder === 'asc') {
                            return valA > valB ? 1 : -1;
                          } else {
                            return valA < valB ? 1 : -1;
                          }
                        });

                      if (processedSnapshots.length === 0 && entry.snapshots.length > 0) {
                        return <div className="text-sm text-notion-text-tertiary italic">No snapshots match the filter "{filterName}"</div>;
                      }

                      return processedSnapshots.map((snapshot) => {
                        const snapshotName = `${modelName}+_${snapshot.iteration}`;
                        const isSelected = selectedSnapshotsForComparison.has(snapshot.id);
                        return (
                          <div
                            key={snapshot.id}
                            className={`flex flex-col gap-3 rounded-lg border p-4 text-sm text-notion-text md:flex-row md:items-center transition ${isSelected
                              ? 'border-notion-blue bg-blue-50'
                              : 'border-notion-border bg-notion-light-gray hover:bg-notion-hover'
                              }`}
                          >
                            <div className="flex items-center gap-3">
                              <input
                                type="checkbox"
                                checked={isSelected}
                                onChange={() => toggleSnapshotSelection(snapshot.id)}
                                className="w-4 h-4 rounded border-notion-border bg-white text-notion-blue focus:ring-notion-blue/50 focus:ring-offset-white cursor-pointer"
                              />
                              <div className="flex-1">
                                <p className="font-mono text-notion-text">{snapshotName}</p>
                                <p className="text-xs text-notion-text-secondary">
                                  Iteration {snapshot.iteration} • {new Date(snapshot.created_at).toLocaleString()}
                                </p>
                                <p className="text-xs text-notion-text-secondary">{snapshot.file_path}</p>
                              </div>
                            </div>
                            <div className="flex flex-wrap gap-4 text-xs text-notion-text ml-7 md:ml-0">
                              {snapshot.metrics_at_save && Object.keys(snapshot.metrics_at_save).length > 0 && (
                                <div>
                                  <p className="text-notion-text-tertiary">Metrics</p>
                                  <p className="font-mono">{Object.keys(snapshot.metrics_at_save).length} saved</p>
                                </div>
                              )}
                              {entry.type === 'Fine-Tuning' && (
                                <div className="flex items-center gap-1 text-amber-600">
                                  <Shield size={14} />
                                  Fine-Tuned
                                </div>
                              )}

                              {/* Metrics Badges */}
                              {snapshot.metrics_at_save && (
                                <div className="flex gap-2">
                                  {snapshot.metrics_at_save.success_rate !== undefined && (
                                    <Badge
                                      label={`Success: ${snapshot.metrics_at_save.success_rate}%`}
                                      tone="bg-notion-light-gray text-emerald-600 border-notion-border"
                                    />
                                  )}
                                  {snapshot.metrics_at_save.failure_rate !== undefined && (
                                    <Badge
                                      label={`Fail: ${snapshot.metrics_at_save.failure_rate}%`}
                                      tone="bg-notion-light-gray text-rose-600 border-notion-border"
                                    />
                                  )}
                                  {snapshot.metrics_at_save.reward !== undefined && (
                                    <Badge
                                      label={`R: ${Number(snapshot.metrics_at_save.reward).toFixed(2)}`}
                                      tone="bg-notion-light-gray text-blue-600 border-notion-border"
                                    />
                                  )}
                                </div>
                              )}
                            </div>
                            <div className="flex gap-2 ml-7 md:ml-0">
                              <button
                                onClick={() => handleViewSnapshot(entry, snapshot)}
                                className="flex items-center gap-1 rounded-lg border border-notion-border bg-white px-3 py-2 text-xs font-medium text-notion-text transition hover:border-notion-blue hover:text-notion-blue"
                              >
                                <Info size={14} /> View
                              </button>
                              <button
                                onClick={() => handleDeleteSnapshot(entry.id, snapshot.id)}
                                className="flex items-center gap-1 rounded-lg border border-red-200 bg-white px-3 py-2 text-xs font-medium text-red-600 transition hover:bg-red-50 hover:border-red-300"
                              >
                                <Trash2 size={14} /> Delete
                              </button>
                            </div>
                          </div>
                        );
                      });
                    })()}
                  </div>
                </Card>
              );
            })}
          </div >
        )
      }

      {
        selectedSnapshot && (
          <TrainingInfoModal
            open={true}
            onClose={() => setSelectedSnapshot(null)}
            experiment={selectedSnapshot.experiment}
            experimentName={selectedSnapshot.experiment.name}
            experimentType={selectedSnapshot.experiment.type}
            experimentId={selectedSnapshot.experiment.id}
            experimentStatus="Completed"
            totalSteps={selectedSnapshot.experiment.total_steps}
            maxStep={selectedSnapshot.snapshot.iteration}
          />
        )
      }


      {
        showCompareModal && (
          <CompareModelsModal
            open={showCompareModal}
            onClose={() => setShowCompareModal(false)}
            snapshots={getSelectedSnapshots()}
          />
        )
      }
    </div >
  );
}
