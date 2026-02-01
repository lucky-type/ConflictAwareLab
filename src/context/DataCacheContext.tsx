import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { api } from '../services/api';
import type { ModelSnapshot, ExperimentProgress, Experiment, ExperimentMetric } from '../services/api';

interface DataCacheContextType {
  // Snapshots cache
  allSnapshots: ModelSnapshot[];
  snapshotsByExperiment: Record<number, ModelSnapshot[]>;

  // Completed experiment metrics cache
  completedMetrics: Record<number, ExperimentProgress>;

  // Static data cache
  experiments: Experiment[];
  agents: any[];
  environments: any[];
  rewardFunctions: any[];
  residualConnectors: any[];

  // Loading states
  isLoadingSnapshots: boolean;
  isLoadingMetrics: boolean;
  isLoadingStaticData: boolean;

  // Refresh methods
  refreshSnapshots: () => Promise<void>;
  refreshMetrics: () => Promise<void>;
  refreshStaticData: () => Promise<Experiment[]>;
  refreshAll: () => Promise<void>;

  // Add new snapshot to cache (when experiment completes)
  addCompletedExperimentMetrics: (experimentId: number) => Promise<void>;
}

const DataCacheContext = createContext<DataCacheContextType | undefined>(undefined);

export function DataCacheProvider({ children }: { children: ReactNode }) {
  const [allSnapshots, setAllSnapshots] = useState<ModelSnapshot[]>([]);
  const [snapshotsByExperiment, setSnapshotsByExperiment] = useState<Record<number, ModelSnapshot[]>>({});
  const [completedMetrics, setCompletedMetrics] = useState<Record<number, ExperimentProgress>>({});
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [agents, setAgents] = useState<any[]>([]);
  const [environments, setEnvironments] = useState<any[]>([]);
  const [rewardFunctions, setRewardFunctions] = useState<any[]>([]);
  const [residualConnectors, setResidualConnectors] = useState<any[]>([]);
  const [isLoadingSnapshots, setIsLoadingSnapshots] = useState(true);
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(true);
  const [isLoadingStaticData, setIsLoadingStaticData] = useState(true);

  // Load all snapshots
  const refreshSnapshots = useCallback(async () => {
    try {
      setIsLoadingSnapshots(true);
      const snapshots = await api.getAllSnapshots();
      setAllSnapshots(snapshots);

      // Group by experiment_id for quick lookup
      const grouped: Record<number, ModelSnapshot[]> = {};
      snapshots.forEach(snapshot => {
        if (!grouped[snapshot.experiment_id]) {
          grouped[snapshot.experiment_id] = [];
        }
        grouped[snapshot.experiment_id].push(snapshot);
      });
      setSnapshotsByExperiment(grouped);

      console.log(`[DataCache] Loaded ${snapshots.length} snapshots`);
    } catch (err) {
      console.error('[DataCache] Failed to load snapshots:', err);
    } finally {
      setIsLoadingSnapshots(false);
    }
  }, []);

  // Load all metrics for completed/cancelled experiments
  // Accepts experiments list to avoid duplicate API call
  const refreshMetrics = useCallback(async (experimentsList?: Experiment[]) => {
    try {
      setIsLoadingMetrics(true);

      // Use provided experiments or fetch if not provided
      const exps = experimentsList ?? await api.getExperiments();
      const completedExps = exps.filter(
        exp => exp.status === 'Completed' || exp.status === 'Cancelled'
      );

      // Use bulk endpoint to fetch latest metrics for all experiments in one request
      const expIds = completedExps.map(exp => exp.id);
      const bulkMetrics = await api.getBulkLatestMetrics(expIds);

      const metricsMap: Record<number, ExperimentProgress> = {};

      // Process the bulk result
      completedExps.forEach(exp => {
        const latestMetric = bulkMetrics[exp.id];
        if (latestMetric) {
          metricsMap[exp.id] = {
            experiment_id: exp.id,
            step: latestMetric.step,
            total_steps: exp.total_steps,
            episode: latestMetric.values.total_episodes,
            total_episodes: exp.evaluation_episodes,
            metrics: latestMetric.values,
            status: exp.status,
            timestamp: Date.now(),
          };
        }
      });

      setCompletedMetrics(metricsMap);
      console.log(`[DataCache] Loaded metrics for ${Object.keys(metricsMap).length} completed experiments (bulk)`);
    } catch (err) {
      console.error('[DataCache] Failed to load metrics:', err);
    } finally {
      setIsLoadingMetrics(false);
    }
  }, []);

  // Load all static data (experiments, agents, environments, etc.)
  const refreshStaticData = useCallback(async () => {
    try {
      setIsLoadingStaticData(true);
      const [exps, agts, envs, rews, conns] = await Promise.all([
        api.getExperiments(),
        api.getAgents(),
        api.getEnvironments(),
        api.getRewardFunctions(),
        api.getResidualConnectors(),
      ]);
      setExperiments(exps);
      setAgents(agts);
      setEnvironments(envs);
      setRewardFunctions(rews);
      setResidualConnectors(conns);
      console.log(`[DataCache] Loaded static data: ${exps.length} experiments, ${agts.length} agents, ${envs.length} environments`);

      // Return experiments so they can be reused
      return exps;
    } catch (err) {
      console.error('[DataCache] Failed to load static data:', err);
      return [];
    } finally {
      setIsLoadingStaticData(false);
    }
  }, []);

  // Add metrics for a single completed experiment
  const addCompletedExperimentMetrics = useCallback(async (experimentId: number) => {
    try {
      const metrics = await api.getExperimentMetrics(experimentId);
      if (metrics.length > 0) {
        const latestMetric = metrics[metrics.length - 1];
        const exp = await api.getExperiment(experimentId);

        const metricsData: ExperimentProgress = {
          experiment_id: experimentId,
          step: latestMetric.step,
          total_steps: exp.total_steps,
          episode: latestMetric.values.total_episodes,
          total_episodes: exp.evaluation_episodes,
          metrics: latestMetric.values,
          status: exp.status,
          timestamp: Date.now(),
        };

        setCompletedMetrics(prev => ({ ...prev, [experimentId]: metricsData }));
        console.log(`[DataCache] Added metrics for experiment ${experimentId}`);
      }
    } catch (err) {
      console.warn(`[DataCache] Failed to add metrics for experiment ${experimentId}:`, err);
    }
  }, []);

  // Refresh all data
  const refreshAll = useCallback(async () => {
    const [, exps] = await Promise.all([
      refreshSnapshots(),
      refreshStaticData()
    ]);
    // Reuse experiments from refreshStaticData to avoid duplicate fetch
    await refreshMetrics(exps);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty deps to avoid circular dependency

  // Initial load on mount - optimized to avoid duplicate experiments fetch
  useEffect(() => {
    console.log('[DataCache] Initializing cache...');
    (async () => {
      const [, exps] = await Promise.all([
        refreshSnapshots(),
        refreshStaticData()
      ]);
      // Reuse experiments to avoid duplicate API call
      await refreshMetrics(exps);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run once on mount

  const value: DataCacheContextType = {
    allSnapshots,
    snapshotsByExperiment,
    completedMetrics,
    experiments,
    agents,
    environments,
    rewardFunctions,
    residualConnectors,
    isLoadingSnapshots,
    isLoadingMetrics,
    isLoadingStaticData,
    refreshSnapshots,
    refreshMetrics,
    refreshStaticData,
    refreshAll,
    addCompletedExperimentMetrics,
  };

  return <DataCacheContext.Provider value={value}>{children}</DataCacheContext.Provider>;
}

export function useDataCache() {
  const context = useContext(DataCacheContext);
  if (!context) {
    throw new Error('useDataCache must be used within DataCacheProvider');
  }
  return context;
}
