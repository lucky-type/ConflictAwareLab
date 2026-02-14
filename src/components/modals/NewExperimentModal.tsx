import { FormEvent, useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { Card, CardHeader } from '../common/Card';
import { api } from '../../services/api';
import type { Agent, RewardFunction, Environment, ModelSnapshot, Experiment, ResidualConnector } from '../../services/api';
import { useDataCache } from '../../context/DataCacheContext';

type HeuristicAlgorithmOption = 'potential_field' | 'vfh_lite';

interface Props {
  open: boolean;
  onClose: () => void;
  onSave?: (experimentId?: number) => void;
}

export default function NewExperimentModal({ open, onClose, onSave }: Props) {
  const {
    allSnapshots,
    agents: cachedAgents,
    rewardFunctions: cachedRewardFunctions,
    environments: cachedEnvironments,
    experiments: cachedExperiments,
    residualConnectors: cachedResidualConnectors,
  } = useDataCache();
  const [tab, setTab] = useState<'Training' | 'Simulation' | 'Fine-Tuning' | 'Residual' | 'Evaluation'>('Training');
  const [experimentName, setExperimentName] = useState('');
  const [totalSteps, setTotalSteps] = useState<number | ''>(1000000);
  const [snapshotInterval, setSnapshotInterval] = useState<number | ''>(50000);
  const [maxEpLength, setMaxEpLength] = useState<number | ''>(2000);
  const [agentId, setAgentId] = useState<number | null>(null);
  const [rewardFunctionId, setRewardFunctionId] = useState<number | null>(null);
  const [environmentId, setEnvironmentId] = useState<number | null>(null);
  const [modelSnapshotId, setModelSnapshotId] = useState<number | null>(null);
  const [baseModelAlgorithm, setBaseModelAlgorithm] = useState<string | null>(null);

  // Residual Learning mode states
  const [trainingMode, setTrainingMode] = useState<'standard' | 'residual'>('standard');
  const [residualBaseModelId, setResidualBaseModelId] = useState<number | null>(null);
  const [residualConnectorId, setResidualConnectorId] = useState<number | null>(null);

  // Evaluation mode states
  const [evaluationEpisodes, setEvaluationEpisodes] = useState<number | ''>(100);
  const [fpsDelay, setFpsDelay] = useState<number | ''>(0);
  const [evaluationResidualMode, setEvaluationResidualMode] = useState(false);
  const [evaluationBaseModelId, setEvaluationBaseModelId] = useState<number | null>(null);
  const [evaluationResidualModelId, setEvaluationResidualModelId] = useState<number | null>(null);
  const [useHeuristicBaseline, setUseHeuristicBaseline] = useState(false);
  const [heuristicAlgorithm, setHeuristicAlgorithm] = useState<HeuristicAlgorithmOption | null>(null);
  const [runWithoutSimulation, setRunWithoutSimulation] = useState(false);

  // Reproducibility
  const [seed, setSeed] = useState<number | null>(null);

  // Safety Constraint (Lagrangian / Risk-aware RL) states
  const [safetyConstraintEnabled, setSafetyConstraintEnabled] = useState(false);
  const [riskBudget, setRiskBudget] = useState<number>(0.02);
  const [initialLambda, setInitialLambda] = useState<number>(0.0);
  const [lambdaLearningRate, setLambdaLearningRate] = useState<number>(0.02);
  const [updateFrequency, setUpdateFrequency] = useState<number>(50);
  const [costSignalPreset, setCostSignalPreset] = useState<string>('balanced');
  const [collisionWeight, setCollisionWeight] = useState<number>(0.1);
  const [nearMissWeight, setNearMissWeight] = useState<number>(0.01);
  const [dangerZoneWeight, setDangerZoneWeight] = useState<number>(0.005);
  const [nearMissThreshold, setNearMissThreshold] = useState<number>(1.5);
  const [ignoreWalls, setIgnoreWalls] = useState<boolean>(true);

  // Wall proximity costs (separate from drone proximity)
  const [wallNearMissWeight, setWallNearMissWeight] = useState<number>(0.005);
  const [wallDangerZoneWeight, setWallDangerZoneWeight] = useState<number>(0.01);
  const [wallNearMissThreshold, setWallNearMissThreshold] = useState<number>(1.0);

  // Lambda warmup schedule states
  const [targetLambda, setTargetLambda] = useState<number | null>(null);
  const [warmupEpisodes, setWarmupEpisodes] = useState<number>(0);
  const [warmupSchedule, setWarmupSchedule] = useState<string>('exponential');

  // Local state mirrors cached data for form interactions
  const [agents, setAgents] = useState<Agent[]>([]);
  const [rewardFunctions, setRewardFunctions] = useState<RewardFunction[]>([]);
  const [environments, setEnvironments] = useState<Environment[]>([]);
  const [snapshots, setSnapshots] = useState<ModelSnapshot[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [residualConnectors, setResidualConnectors] = useState<ResidualConnector[]>([]);
  const [loading, setLoading] = useState(false);

  // Auto-populate cost weights based on preset
  const applyCostPreset = (preset: string) => {
    switch (preset) {
      case 'balanced':
        setCollisionWeight(0.1);
        setNearMissWeight(0.01);
        setDangerZoneWeight(0.005);
        break;
      case 'strict_safety':
        setCollisionWeight(0.2);
        setNearMissWeight(0.05);
        setDangerZoneWeight(0.02);
        break;
      case 'near_miss_focused':
        setCollisionWeight(0.1);
        setNearMissWeight(0.05);
        setDangerZoneWeight(0.001);
        break;
      case 'collision_only':
        setCollisionWeight(0.1);
        setNearMissWeight(0.0);
        setDangerZoneWeight(0.0);
        break;
    }
  };

  // Auto-apply preset when costSignalPreset changes
  useEffect(() => {
    applyCostPreset(costSignalPreset);
  }, [costSignalPreset]);

  useEffect(() => {
    if (open) {
      loadData();
    }
  }, [open]);

  // Auto-select agent and reward function when snapshot changes in Fine-Tuning mode
  useEffect(() => {
    if (tab === 'Fine-Tuning' && modelSnapshotId) {
      const selectedSnapshot = snapshots.find(s => s.id === modelSnapshotId);
      if (selectedSnapshot?.metrics_at_save) {
        const agentIdFromSnapshot = selectedSnapshot.metrics_at_save.agent_id;
        const rewardIdFromSnapshot = selectedSnapshot.metrics_at_save.reward_id;

        if (agentIdFromSnapshot) {
          setAgentId(agentIdFromSnapshot);

          // Find agent to get algorithm type
          const agent = agents.find(a => a.id === agentIdFromSnapshot);
          if (agent) {
            setBaseModelAlgorithm(agent.type);
          }
        }
        if (rewardIdFromSnapshot) {
          setRewardFunctionId(rewardIdFromSnapshot);
        }
      }
    }
  }, [tab, modelSnapshotId, snapshots, agents]);

  // Auto-select agent and reward function when base model changes in Residual tab
  useEffect(() => {
    if (tab === 'Residual' && residualBaseModelId) {
      const selectedSnapshot = snapshots.find(s => s.id === residualBaseModelId);
      if (selectedSnapshot?.metrics_at_save) {
        const agentIdFromSnapshot = selectedSnapshot.metrics_at_save.agent_id;
        const rewardIdFromSnapshot = selectedSnapshot.metrics_at_save.reward_id;

        if (agentIdFromSnapshot) {
          setAgentId(agentIdFromSnapshot);
        }
        if (rewardIdFromSnapshot) {
          setRewardFunctionId(rewardIdFromSnapshot);
        }
      }
    }
  }, [tab, residualBaseModelId, snapshots]);

  // Enable safety constraint automatically for Residual tab
  useEffect(() => {
    if (tab === 'Residual') {
      setSafetyConstraintEnabled(true);
    } else {
      // For Training/Fine-Tuning, default to false unless user had it checked (but resetting on tab switch is cleaner)
      setSafetyConstraintEnabled(false);
    }
  }, [tab]);

  // Auto-select residual connector and base model when residual model is selected in Evaluation mode
  useEffect(() => {
    if (tab === 'Evaluation' && evaluationResidualMode && evaluationResidualModelId) {
      const residualSnapshot = snapshots.find(s => s.id === evaluationResidualModelId);
      if (residualSnapshot?.metrics_at_save) {
        // Auto-select residual connector from snapshot metadata
        const connectorIdFromSnapshot = residualSnapshot.metrics_at_save.residual_connector_id;
        if (connectorIdFromSnapshot && !residualConnectorId) {
          setResidualConnectorId(connectorIdFromSnapshot);
        }

        // Auto-select base model from snapshot metadata if not already set
        const baseModelIdFromSnapshot = residualSnapshot.metrics_at_save.residual_base_model_id;
        if (baseModelIdFromSnapshot && !evaluationBaseModelId) {
          setEvaluationBaseModelId(baseModelIdFromSnapshot);
        }
      }
    }
  }, [tab, evaluationResidualMode, evaluationResidualModelId, snapshots]);

  const loadData = () => {
    try {
      // Use cached data instead of API calls
      setAgents(cachedAgents);
      setRewardFunctions(cachedRewardFunctions);
      setEnvironments(cachedEnvironments);
      setExperiments(cachedExperiments);
      setResidualConnectors(cachedResidualConnectors);
      setSnapshots(allSnapshots);

      // Set defaults - always set if data is available and current selection is invalid
      if (cachedAgents.length > 0 && !agentId) setAgentId(cachedAgents[0].id);
      if (cachedRewardFunctions.length > 0 && !rewardFunctionId) setRewardFunctionId(cachedRewardFunctions[0].id);

      // Always set environment to first available if none selected
      if (cachedEnvironments.length > 0 && (!environmentId || !cachedEnvironments.find(e => e.id === environmentId))) {
        setEnvironmentId(cachedEnvironments[0].id);
        console.log('Setting environmentId to:', cachedEnvironments[0].id, cachedEnvironments[0].name);
      }

      if (allSnapshots.length > 0 && !modelSnapshotId) setModelSnapshotId(allSnapshots[0].id);

      console.log('NewExperimentModal: Data loaded from cache', {
        agents: cachedAgents.length,
        environments: cachedEnvironments.length,
        rewardFunctions: cachedRewardFunctions.length,
        currentEnvironmentId: environmentId,
        willSetEnvironmentId: cachedEnvironments.length > 0 && (!environmentId || !cachedEnvironments.find(e => e.id === environmentId)) ? cachedEnvironments[0]?.id : 'no change'
      });
    } catch (err) {
      console.error('Failed to load data:', err);
      alert('Failed to load data for experiment creation');
    }
  };

  if (!open) return null;

  const submit = async (event: FormEvent) => {
    event.preventDefault();

    // Check if data is still loading
    if (environments.length === 0) {
      alert('Please wait for data to load');
      return;
    }

    if (!environmentId) {
      alert('Please select an environment');
      console.error('Submit failed - no environmentId:', { environmentId, environments: environments.length });
      return;
    }

    console.log('Submitting experiment with environmentId:', environmentId);

    try {
      setLoading(true);

      const finalName = experimentName.trim() || `${tab}_${Date.now()}`;

      let createdExperiment;
      if (tab === 'Training') {
        if (!agentId || !rewardFunctionId) {
          alert('Please select agent and reward function');
          return;
        }

        const trainingConfig: any = {
          name: finalName,
          type: 'Training',
          env_id: environmentId,
          agent_id: agentId,
          reward_id: rewardFunctionId,
          total_steps: typeof totalSteps === 'number' ? totalSteps : parseInt(String(totalSteps)) || 1000000,
          snapshot_freq: typeof snapshotInterval === 'number' ? snapshotInterval : parseInt(String(snapshotInterval)) || 50000,
          max_ep_length: typeof maxEpLength === 'number' ? maxEpLength : parseInt(String(maxEpLength)) || 2000,
          training_mode: 'standard',
          run_without_simulation: runWithoutSimulation,
          seed: seed,
        };

        // Add safety constraint parameters if enabled (Lagrangian)
        if (safetyConstraintEnabled) {
          trainingConfig.safety_constraint = {
            enabled: safetyConstraintEnabled,
            risk_budget: riskBudget,
            initial_lambda: initialLambda,
            lambda_learning_rate: lambdaLearningRate,
            update_frequency: updateFrequency,
            target_lambda: targetLambda,
            warmup_episodes: warmupEpisodes,
            warmup_schedule: warmupSchedule,
            cost_signal_preset: costSignalPreset,
            collision_weight: collisionWeight,
            near_miss_weight: nearMissWeight,
            danger_zone_weight: dangerZoneWeight,
            near_miss_threshold: nearMissThreshold,
            ignore_walls: ignoreWalls,
            wall_near_miss_weight: wallNearMissWeight,
            wall_danger_zone_weight: wallDangerZoneWeight,
            wall_near_miss_threshold: wallNearMissThreshold,
          };
        }

        createdExperiment = await api.createExperiment(trainingConfig);
      } else if (tab === 'Residual') {
        if (!agentId || !rewardFunctionId) {
          alert('Please select agent and reward function');
          return;
        }
        if (!residualBaseModelId) {
          alert('Please select a base model for residual learning');
          return;
        }
        if (!residualConnectorId) {
          alert('Please select a residual connector');
          return;
        }

        const residualConfig: any = {
          name: finalName,
          type: 'Training',
          env_id: environmentId,
          agent_id: agentId,
          reward_id: rewardFunctionId,
          total_steps: typeof totalSteps === 'number' ? totalSteps : parseInt(String(totalSteps)) || 1000000,
          snapshot_freq: typeof snapshotInterval === 'number' ? snapshotInterval : parseInt(String(snapshotInterval)) || 50000,
          max_ep_length: typeof maxEpLength === 'number' ? maxEpLength : parseInt(String(maxEpLength)) || 2000,
          training_mode: 'residual',
          residual_base_model_id: residualBaseModelId,
          residual_connector_id: residualConnectorId,
          run_without_simulation: runWithoutSimulation,
          seed: seed,
        };

        // Safety constraint is mandatory for residual learning
        residualConfig.safety_constraint = {
          enabled: safetyConstraintEnabled || tab === 'Residual',
          risk_budget: riskBudget,
          initial_lambda: initialLambda,
          lambda_learning_rate: lambdaLearningRate,
          update_frequency: updateFrequency,
          cost_signal_preset: costSignalPreset,
          near_miss_threshold: nearMissThreshold,
          collision_weight: collisionWeight,
          near_miss_weight: nearMissWeight,
          danger_zone_weight: dangerZoneWeight,
          ignore_walls: ignoreWalls,
          wall_near_miss_weight: wallNearMissWeight,
          wall_danger_zone_weight: wallDangerZoneWeight,
          wall_near_miss_threshold: wallNearMissThreshold,
          target_lambda: targetLambda,
          warmup_episodes: warmupEpisodes,
          warmup_schedule: warmupSchedule,
        };

        createdExperiment = await api.createExperiment(residualConfig);
      } else if (tab === 'Fine-Tuning') {
        if (!agentId || !rewardFunctionId) {
          alert('Please select agent and reward function');
          return;
        }
        if (!modelSnapshotId) {
          alert('Please select a base model snapshot');
          return;
        }

        const fineTuningConfig: any = {
          name: finalName,
          type: 'Fine-Tuning',
          env_id: environmentId,
          agent_id: agentId,
          reward_id: rewardFunctionId,
          total_steps: typeof totalSteps === 'number' ? totalSteps : parseInt(String(totalSteps)) || 1000000,
          snapshot_freq: typeof snapshotInterval === 'number' ? snapshotInterval : parseInt(String(snapshotInterval)) || 50000,
          max_ep_length: typeof maxEpLength === 'number' ? maxEpLength : parseInt(String(maxEpLength)) || 2000,
          base_model_snapshot_id: modelSnapshotId,
          fine_tuning_strategy: 'full_finetune',
          run_without_simulation: runWithoutSimulation,
          seed: seed,
        };

        // Add safety constraint parameters if enabled (Lagrangian)
        if (safetyConstraintEnabled) {
          fineTuningConfig.safety_constraint = {
            enabled: safetyConstraintEnabled,
            risk_budget: riskBudget,
            initial_lambda: initialLambda,
            lambda_learning_rate: lambdaLearningRate,
            update_frequency: updateFrequency,
            target_lambda: targetLambda,
            warmup_episodes: warmupEpisodes,
            warmup_schedule: warmupSchedule,
            cost_signal_preset: costSignalPreset,
            collision_weight: collisionWeight,
            near_miss_weight: nearMissWeight,
            danger_zone_weight: dangerZoneWeight,
            near_miss_threshold: nearMissThreshold,
            ignore_walls: ignoreWalls,
            wall_near_miss_weight: wallNearMissWeight,
            wall_danger_zone_weight: wallDangerZoneWeight,
            wall_near_miss_threshold: wallNearMissThreshold,
          };
        }

        createdExperiment = await api.createExperiment(fineTuningConfig);
      } else if (tab === 'Evaluation') {
        if (useHeuristicBaseline) {
          if (!heuristicAlgorithm) {
            alert('Please select one heuristic algorithm');
            return;
          }
        } else {
          if (!evaluationBaseModelId) {
            alert('Please select a base model');
            return;
          }

          if (evaluationResidualMode && !evaluationResidualModelId) {
            alert('Please select a residual model when residual mode is enabled');
            return;
          }
        }

        const evaluationConfig: any = {
          name: finalName,
          type: 'Evaluation',
          env_id: environmentId,
          agent_id: agentId || agents[0]?.id || 1, // Use placeholder if needed
          reward_id: rewardFunctionId || rewardFunctions[0]?.id || 1, // Use placeholder if needed
          total_steps: 1, // Minimum to pass validation, not used for evaluation
          snapshot_freq: 1, // Minimum to pass validation, not used for evaluation
          max_ep_length: typeof maxEpLength === 'number' ? maxEpLength : parseInt(String(maxEpLength)) || 2000,
          evaluation_episodes: typeof evaluationEpisodes === 'number' ? evaluationEpisodes : parseInt(String(evaluationEpisodes)) || 100,
          fps_delay: typeof fpsDelay === 'number' ? fpsDelay : parseInt(String(fpsDelay)) || 0,
          run_without_simulation: runWithoutSimulation,
          seed: seed,
        };

        // Set policy configuration for evaluation mode
        if (useHeuristicBaseline) {
          evaluationConfig.training_mode = 'standard';
          evaluationConfig.evaluation_policy_mode = 'heuristic';
          evaluationConfig.heuristic_algorithm = heuristicAlgorithm;
        } else {
          evaluationConfig.evaluation_policy_mode = 'model';
          evaluationConfig.heuristic_algorithm = null;

          if (evaluationResidualMode) {
            evaluationConfig.training_mode = 'residual';
            evaluationConfig.residual_base_model_id = evaluationBaseModelId;
            evaluationConfig.model_snapshot_id = evaluationResidualModelId;
            evaluationConfig.residual_connector_id = residualConnectorId; // May need to add selector for this
          } else {
            evaluationConfig.training_mode = 'standard';
            evaluationConfig.model_snapshot_id = evaluationBaseModelId;
          }
        }

        createdExperiment = await api.createExperiment(evaluationConfig);
      } else {
        if (!modelSnapshotId) {
          alert('Please select a trained model snapshot');
          return;
        }

        // For simulation, we still need agent and reward IDs from the backend perspective
        // Use the first available ones as placeholders
        createdExperiment = await api.createExperiment({
          name: finalName,
          type: 'Simulation',
          env_id: environmentId,
          agent_id: agentId || agents[0]?.id || 1,
          reward_id: rewardFunctionId || rewardFunctions[0]?.id || 1,
          total_steps: 1, // Minimum 1 to pass validation
          snapshot_freq: 1, // Minimum 1 to pass validation
          max_ep_length: typeof maxEpLength === 'number' ? maxEpLength : parseInt(String(maxEpLength)) || 2000,
          model_snapshot_id: modelSnapshotId,
          seed: seed,
        });
      }

      // Auto-start the experiment (check capacity first)
      if (createdExperiment && createdExperiment.id) {
        try {
          // Check job capacity before auto-starting
          const capacity = await api.getJobCapacity();
          if (!capacity.can_start_new) {
            alert(`Experiment created but not started: Maximum concurrent jobs (${capacity.max_concurrent_jobs}) reached. ${capacity.available_slots} slots available. Please wait for a job to complete or manually start later.`);
          } else {
            await api.startExperiment(createdExperiment.id);
          }
        } catch (startErr) {
          console.error('Failed to start experiment:', startErr);
          // Don't fail the whole operation if start fails
        }
      }

      onSave?.(createdExperiment?.id);
      setExperimentName('');
      onClose();
    } catch (err) {
      alert('Failed to create experiment: ' + (err instanceof Error ? err.message : 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
    >
      <div className="w-full max-w-2xl overflow-y-auto max-h-[90vh] rounded-lg border border-notion-border bg-white shadow-xl" onClick={(e) => e.stopPropagation()}>
        <div className="p-6">
          <div className="flex items-start justify-between">
            <CardHeader title="New Experiment" />
            <button onClick={onClose} className="text-notion-text-secondary hover:text-notion-text transition-colors">
              <X size={20} />
            </button>
          </div>

          <div className="mb-4 grid grid-cols-5 rounded-lg border border-notion-border bg-notion-light-gray text-sm font-medium text-notion-text-secondary">
            <button
              className={`rounded-lg px-4 py-2 transition ${tab === 'Training' ? 'bg-notion-blue/10 text-notion-blue font-medium' : 'hover:bg-notion-hover text-notion-text-secondary hover:text-notion-text'
                }`}
              onClick={() => setTab('Training')}
              type="button"
            >
              Training
            </button>
            <button
              className={`rounded-lg px-4 py-2 transition ${tab === 'Residual' ? 'bg-notion-purple/10 text-notion-purple font-medium' : 'hover:bg-notion-hover text-notion-text-secondary hover:text-notion-text'
                }`}
              onClick={() => setTab('Residual')}
              type="button"
            >
              Residual
            </button>
            <button
              className={`rounded-lg px-4 py-2 transition ${tab === 'Fine-Tuning' ? 'bg-orange-500/10 text-orange-600 font-medium' : 'hover:bg-notion-hover text-notion-text-secondary hover:text-notion-text'
                }`}
              onClick={() => setTab('Fine-Tuning')}
              type="button"
            >
              Fine-Tuning
            </button>
            <button
              className={`rounded-lg px-4 py-2 transition ${tab === 'Simulation' ? 'bg-notion-blue/10 text-notion-blue font-medium' : 'hover:bg-notion-hover text-notion-text-secondary hover:text-notion-text'
                }`}
              onClick={() => setTab('Simulation')}
              type="button"
            >
              Simulation
            </button>
            <button
              className={`rounded-lg px-4 py-2 transition ${tab === 'Evaluation' ? 'bg-green-500/10 text-green-600 font-medium' : 'hover:bg-notion-hover text-notion-text-secondary hover:text-notion-text'
                }`}
              onClick={() => setTab('Evaluation')}
              type="button"
            >
              Evaluation
            </button>
          </div>

          <form onSubmit={submit} className="grid gap-6 md:grid-cols-2">
            <div className="space-y-4">
              <label className="space-y-2">
                <span className="text-notion-text">Experiment Name</span>
                <input
                  type="text"
                  value={experimentName}
                  onChange={(e) => setExperimentName(e.target.value)}
                  placeholder={`Auto-generated if empty`}
                  className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                />
              </label>

              {tab === 'Training' ? (
                <>
                  <label className="block space-y-2 text-sm">
                    <span className="text-notion-text">Agent</span>
                    <select
                      value={agentId ?? ''}
                      onChange={(e) => setAgentId(parseInt(e.target.value))}
                      className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                      required
                    >
                      {agents.length === 0 ? (
                        <option value="">No agents available</option>
                      ) : (
                        agents.map((agent) => (
                          <option key={agent.id} value={agent.id}>
                            {agent.name} ({agent.type})
                          </option>
                        ))
                      )}
                    </select>
                  </label>

                  <label className="block space-y-2 text-sm">
                    <span className="text-notion-text">Reward function</span>
                    <select
                      value={rewardFunctionId ?? ''}
                      onChange={(e) => setRewardFunctionId(parseInt(e.target.value))}
                      className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                      required
                    >
                      {rewardFunctions.length === 0 ? (
                        <option value="">No reward functions available</option>
                      ) : (
                        rewardFunctions.map((reward) => (
                          <option key={reward.id} value={reward.id}>
                            {reward.name}
                          </option>
                        ))
                      )}
                    </select>
                  </label>

                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <label className="space-y-2">
                      <span className="text-notion-text">Total steps</span>
                      <input
                        type="number"
                        value={totalSteps}
                        step="1000"
                        onChange={(e) => setTotalSteps(e.target.value === '' ? '' : parseInt(e.target.value))}
                        className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                        required
                      />
                    </label>
                    <label className="space-y-2">
                      <span className="text-notion-text">Snapshot freq</span>
                      <input
                        type="number"
                        value={snapshotInterval}
                        step="100"
                        onChange={(e) => setSnapshotInterval(e.target.value === '' ? '' : parseInt(e.target.value))}
                        className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                        required
                      />
                    </label>
                  </div>
                </>
              ) : tab === 'Fine-Tuning' ? (
                <>
                  <label className="block space-y-2 text-sm">
                    <span className="text-notion-text">Base model (snapshot)</span>
                    <select
                      value={modelSnapshotId ?? ''}
                      onChange={(e) => setModelSnapshotId(parseInt(e.target.value))}
                      className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                      required
                    >
                      {snapshots.length === 0 ? (
                        <option value="">No snapshots available</option>
                      ) : (
                        snapshots.map((snapshot) => (
                          <option key={snapshot.id} value={snapshot.id}>
                            Snapshot #{snapshot.id} (Exp {snapshot.experiment_id}) - Iter {snapshot.iteration}
                          </option>
                        ))
                      )}
                    </select>
                    <p className="text-xs text-notion-text-tertiary">
                      Choose a trained model checkpoint to continue training from.
                    </p>
                  </label>

                  <div className="rounded-lg bg-orange-950/20 border border-orange-500/20 px-3 py-2 mb-4">
                    <p className="text-xs text-orange-400 font-medium mb-1">
                      ⚡ Full Fine-Tuning
                    </p>
                    <p className="text-xs text-notion-text leading-relaxed">
                      All model parameters will be trained.
                    </p>
                  </div>

                  <label className="block space-y-2 text-sm">
                    <span className="text-notion-text">
                      Agent
                      <span className="ml-2 text-xs text-notion-text-tertiary">(from snapshot)</span>
                    </span>
                    <select
                      value={agentId ?? ''}
                      onChange={(e) => setAgentId(parseInt(e.target.value))}
                      className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none disabled:opacity-60 disabled:cursor-not-allowed"
                      required
                      disabled={true}
                    >
                      {agents.length === 0 ? (
                        <option value="">No agents available</option>
                      ) : (
                        agents.map((agent) => (
                          <option key={agent.id} value={agent.id}>
                            {agent.name} ({agent.type})
                          </option>
                        ))
                      )}
                    </select>
                  </label>

                  <label className="block space-y-2 text-sm">
                    <span className="text-notion-text">
                      Reward function
                      <span className="ml-2 text-xs text-notion-text-tertiary">(from snapshot)</span>
                    </span>
                    <select
                      value={rewardFunctionId ?? ''}
                      onChange={(e) => setRewardFunctionId(parseInt(e.target.value))}
                      className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none disabled:opacity-60 disabled:cursor-not-allowed"
                      required
                      disabled={true}
                    >
                      {rewardFunctions.length === 0 ? (
                        <option value="">No reward functions available</option>
                      ) : (
                        rewardFunctions.map((reward) => (
                          <option key={reward.id} value={reward.id}>
                            {reward.name}
                          </option>
                        ))
                      )}
                    </select>
                  </label>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <label className="space-y-2">
                      <span className="text-notion-text">Total steps</span>
                      <input
                        type="number"
                        value={totalSteps}
                        step="1000"
                        onChange={(e) => setTotalSteps(e.target.value === '' ? '' : parseInt(e.target.value))}
                        className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                        required
                      />
                    </label>
                    <label className="space-y-2">
                      <span className="text-notion-text">Snapshot freq</span>
                      <input
                        type="number"
                        value={snapshotInterval}
                        step="100"
                        onChange={(e) => setSnapshotInterval(e.target.value === '' ? '' : parseInt(e.target.value))}
                        className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                        required
                      />
                    </label>
                  </div>
                </>
              ) : tab === 'Residual' ? (
                <>
                  <label className="block space-y-2 text-sm">
                    <span className="text-notion-text">Base Model (Frozen)</span>
                    <select
                      value={residualBaseModelId ?? ''}
                      onChange={(e) => setResidualBaseModelId(parseInt(e.target.value))}
                      className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                      required
                    >
                      <option value="">Select a base model...</option>
                      {snapshots.length === 0 ? (
                        <option value="" disabled>No snapshots available</option>
                      ) : (
                        snapshots.map((snapshot) => {
                          const experiment = experiments.find(e => e.id === snapshot.experiment_id);
                          return (
                            <option key={snapshot.id} value={snapshot.id}>
                              {experiment?.name || `Exp ${snapshot.experiment_id}`} - Iteration {snapshot.iteration}
                            </option>
                          );
                        })
                      )}
                    </select>
                    <div className="rounded-lg bg-notion-hover border border-notion-border px-3 py-2">
                      <p className="text-xs text-notion-blue font-medium mb-1">
                        🎯 Residual Learning Mode:
                      </p>
                      <p className="text-xs text-notion-text leading-relaxed">
                        Base model stays frozen. A lightweight residual policy learns delta actions:
                        <span className="text-amber-400"> final_action = base_action + residual_correction</span>
                      </p>
                      <p className="text-xs text-notion-text-secondary mt-2">
                        Use case: Add safety corrections to an existing fast but unsafe policy.
                      </p>
                    </div>
                  </label>

                  <label className="block space-y-2 text-sm">
                    <span className="text-notion-text">Residual Connector</span>
                    <select
                      value={residualConnectorId ?? ''}
                      onChange={(e) => setResidualConnectorId(parseInt(e.target.value))}
                      className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                      required
                    >
                      <option value="">Select a residual connector...</option>
                      {residualConnectors.length === 0 ? (
                        <option value="" disabled>No residual connectors available</option>
                      ) : (
                        residualConnectors.map((connector) => (
                          <option key={connector.id} value={connector.id}>
                            {connector.name} ({connector.algorithm})
                          </option>
                        ))
                      )}
                    </select>
                    <p className="text-xs text-notion-text-tertiary">
                      Defines the residual policy architecture and learning algorithm.
                    </p>
                  </label>

                  <label className="block space-y-2 text-sm">
                    <span className="text-notion-text">
                      Agent
                      <span className="ml-2 text-xs text-notion-text-tertiary">(from base model)</span>
                    </span>
                    <select
                      value={agentId ?? ''}
                      onChange={(e) => setAgentId(parseInt(e.target.value))}
                      className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none disabled:opacity-60 disabled:cursor-not-allowed"
                      required
                      disabled={true}
                    >
                      {agents.length === 0 ? (
                        <option value="">No agents available</option>
                      ) : (
                        agents.map((agent) => (
                          <option key={agent.id} value={agent.id}>
                            {agent.name} ({agent.type})
                          </option>
                        ))
                      )}
                    </select>
                  </label>

                  <label className="block space-y-2 text-sm">
                    <span className="text-notion-text">
                      Reward function
                      <span className="ml-2 text-xs text-notion-text-tertiary">(from base model)</span>
                    </span>
                    <select
                      value={rewardFunctionId ?? ''}
                      onChange={(e) => setRewardFunctionId(parseInt(e.target.value))}
                      className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none disabled:opacity-60 disabled:cursor-not-allowed"
                      required
                      disabled={true}
                    >
                      {rewardFunctions.length === 0 ? (
                        <option value="">No reward functions available</option>
                      ) : (
                        rewardFunctions.map((reward) => (
                          <option key={reward.id} value={reward.id}>
                            {reward.name}
                          </option>
                        ))
                      )}
                    </select>
                  </label>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <label className="space-y-2">
                      <span className="text-notion-text">Total steps</span>
                      <input
                        type="number"
                        value={totalSteps}
                        step="1000"
                        onChange={(e) => setTotalSteps(e.target.value === '' ? '' : parseInt(e.target.value))}
                        className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                        required
                      />
                    </label>
                    <label className="space-y-2">
                      <span className="text-notion-text">Snapshot freq</span>
                      <input
                        type="number"
                        value={snapshotInterval}
                        step="100"
                        onChange={(e) => setSnapshotInterval(e.target.value === '' ? '' : parseInt(e.target.value))}
                        className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                        required
                      />
                    </label>
                  </div>
                </>
              ) : tab === 'Evaluation' ? (
                <>
                  <div className="space-y-4">
                    <div className="rounded-lg bg-notion-hover border border-notion-border px-3 py-2 mb-4">
                      <p className="text-xs text-green-400 font-medium mb-1">
                        📊 Evaluation Mode
                      </p>
                      <p className="text-xs text-notion-text leading-relaxed">
                        Run deterministic evaluation for a fixed number of episodes. Records trajectory for replay and shows all metrics like training mode.
                      </p>
                    </div>

                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={useHeuristicBaseline}
                        onChange={(e) => {
                          const enabled = e.target.checked;
                          setUseHeuristicBaseline(enabled);
                          if (enabled) {
                            setEvaluationResidualMode(false);
                            setEvaluationResidualModelId(null);
                          }
                        }}
                        className="rounded border-slate-600 bg-slate-900/80 text-notion-blue focus:ring-cyan-400"
                      />
                      <span className="text-notion-text text-sm">Use Heuristic Baseline</span>
                    </label>

                    {useHeuristicBaseline ? (
                      <div className="space-y-3 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-3">
                        <p className="text-xs font-medium text-emerald-700">
                          Heuristic mode runs observation-only baseline controllers without model snapshots.
                        </p>
                        <label className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            checked={heuristicAlgorithm === 'potential_field'}
                            onChange={(e) => setHeuristicAlgorithm(e.target.checked ? 'potential_field' : null)}
                            className="rounded border-slate-600 bg-slate-900/80 text-notion-blue focus:ring-cyan-400"
                          />
                          <span className="text-notion-text text-sm">Potential Field</span>
                        </label>
                        <label className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            checked={heuristicAlgorithm === 'vfh_lite'}
                            onChange={(e) => setHeuristicAlgorithm(e.target.checked ? 'vfh_lite' : null)}
                            className="rounded border-slate-600 bg-slate-900/80 text-notion-blue focus:ring-cyan-400"
                          />
                          <span className="text-notion-text text-sm">VFH-Lite</span>
                        </label>
                        <p className="text-xs text-notion-text-tertiary">
                          Select exactly one heuristic algorithm.
                        </p>
                      </div>
                    ) : (
                      <>
                        <label className="block space-y-2 text-sm">
                          <span className="text-notion-text">Base Model</span>
                          <select
                            value={evaluationBaseModelId ?? ''}
                            onChange={(e) => setEvaluationBaseModelId(parseInt(e.target.value))}
                            className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                            required
                          >
                            <option value="">Select base model...</option>
                            {snapshots.length === 0 ? (
                              <option value="" disabled>No snapshots available</option>
                            ) : (
                              snapshots.map((snapshot) => {
                                const experiment = experiments.find(e => e.id === snapshot.experiment_id);
                                return (
                                  <option key={snapshot.id} value={snapshot.id}>
                                    {experiment?.name || `Exp ${snapshot.experiment_id}`} - Iteration {snapshot.iteration}
                                  </option>
                                );
                              })
                            )}
                          </select>
                        </label>

                        <label className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            checked={evaluationResidualMode}
                            onChange={(e) => setEvaluationResidualMode(e.target.checked)}
                            className="rounded border-slate-600 bg-slate-900/80 text-notion-blue focus:ring-cyan-400"
                          />
                          <span className="text-notion-text text-sm">Use Residual Policy</span>
                        </label>

                        {evaluationResidualMode && (
                          <>
                            <label className="block space-y-2 text-sm">
                              <span className="text-notion-text">Residual Model</span>
                              <select
                                value={evaluationResidualModelId ?? ''}
                                onChange={(e) => setEvaluationResidualModelId(parseInt(e.target.value))}
                                className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                                required={evaluationResidualMode}
                              >
                                <option value="">Select residual model...</option>
                                {snapshots.length === 0 ? (
                                  <option value="" disabled>No snapshots available</option>
                                ) : (
                                  snapshots
                                    .filter(s => s.metrics_at_save?.training_mode === 'residual')
                                    .map((snapshot) => {
                                      const experiment = experiments.find(e => e.id === snapshot.experiment_id);
                                      return (
                                        <option key={snapshot.id} value={snapshot.id}>
                                          {experiment?.name || `Exp ${snapshot.experiment_id}`} - Iteration {snapshot.iteration} [Residual]
                                        </option>
                                      );
                                    })
                                )}
                              </select>
                              <p className="text-xs text-notion-text-tertiary">
                                Select the trained residual policy snapshot.
                              </p>
                              {evaluationResidualModelId && (() => {
                                const snap = snapshots.find(s => s.id === evaluationResidualModelId);
                                if (snap?.metrics_at_save?.residual_connector_id) {
                                  const connector = residualConnectors.find(c => c.id === snap.metrics_at_save.residual_connector_id);
                                  return (
                                    <div className="text-xs text-notion-text-tertiary bg-notion-hover rounded px-2 py-1 mt-1">
                                      Training used: {connector?.name || `Connector #${snap.metrics_at_save.residual_connector_id}`}
                                    </div>
                                  );
                                }
                                return null;
                              })()}
                            </label>

                            <label className="block space-y-2 text-sm">
                              <span className="text-notion-text">Residual Connector</span>
                              <select
                                value={residualConnectorId ?? ''}
                                onChange={(e) => setResidualConnectorId(parseInt(e.target.value))}
                                className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                                required={evaluationResidualMode}
                              >
                                <option value="">Select residual connector...</option>
                                {residualConnectors.length === 0 ? (
                                  <option value="" disabled>No residual connectors available</option>
                                ) : (
                                  residualConnectors.map((connector) => (
                                    <option key={connector.id} value={connector.id}>
                                      {connector.name} ({connector.algorithm})
                                    </option>
                                  ))
                                )}
                              </select>
                              <p className="text-xs text-notion-text-tertiary">
                                Auto-selected from residual model snapshot. Can be changed if needed.
                              </p>
                              {evaluationResidualModelId && residualConnectorId && (() => {
                                const snap = snapshots.find(s => s.id === evaluationResidualModelId);
                                const trainingConnectorId = snap?.metrics_at_save?.residual_connector_id;
                                if (trainingConnectorId && trainingConnectorId !== residualConnectorId) {
                                  return (
                                    <div className="text-xs text-amber-400 bg-amber-950/20 border border-amber-500/30 rounded px-2 py-1 mt-1">
                                      Warning: Selected connector differs from training. Results may not match expected behavior.
                                    </div>
                                  );
                                }
                                return null;
                              })()}
                            </label>
                          </>
                        )}
                      </>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <label className="space-y-2">
                      <span className="text-notion-text">Episodes</span>
                      <input
                        type="number"
                        value={evaluationEpisodes}
                        min="1"
                        onChange={(e) => setEvaluationEpisodes(e.target.value === '' ? '' : parseInt(e.target.value))}
                        className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                        required
                      />
                      <p className="text-xs text-notion-text-tertiary">Number of episodes to run</p>
                    </label>
                    <label className="space-y-2">
                      <span className="text-notion-text">FPS Delay (ms)</span>
                      <input
                        type="number"
                        value={fpsDelay}
                        min="0"
                        onChange={(e) => setFpsDelay(e.target.value === '' ? '' : parseInt(e.target.value))}
                        className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                      />
                      <p className="text-xs text-notion-text-tertiary">Delay between steps for visualization (0 = no delay)</p>
                    </label>
                  </div>
                </>
              ) : (
                // Simulation tab
                <>
                  <label className="block space-y-2 text-sm">
                    <span className="text-notion-text">Model Snapshot</span>
                    <select
                      value={modelSnapshotId ?? ''}
                      onChange={(e) => setModelSnapshotId(parseInt(e.target.value))}
                      className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                      required
                    >
                      {snapshots.length === 0 ? (
                        <option value="">No snapshots available</option>
                      ) : (
                        snapshots.map((snapshot) => (
                          <option key={snapshot.id} value={snapshot.id}>
                            Snapshot #{snapshot.id} (Exp {snapshot.experiment_id}) - Iter {snapshot.iteration}
                          </option>
                        ))
                      )}
                    </select>
                  </label>
                  <div className="rounded-lg bg-notion-hover border border-notion-border px-3 py-2">
                    <p className="text-xs text-notion-blue font-medium mb-1">
                      Run Simulation
                    </p>
                    <p className="text-xs text-notion-text">
                      Visualize how the trained model performs in the selected environment.
                    </p>
                  </div>
                </>
              )}
            </div>

            <div className="space-y-5">
              <label className="block space-y-2">
                <span className="text-notion-text">Environment</span>
                <select
                  value={environmentId ?? ''}
                  onChange={(e) => setEnvironmentId(parseInt(e.target.value))}
                  className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                  required
                >
                  {environments.length === 0 ? (
                    <option value="">No environments available</option>
                  ) : (
                    environments.map((env) => (
                      <option key={env.id} value={env.id}>
                        {env.name}
                      </option>
                    ))
                  )}
                </select>
              </label>

              {/* Seed for reproducibility */}
              <label className="block space-y-2">
                <span className="text-notion-text">
                  Seed
                  <span className="ml-2 text-xs text-notion-text-tertiary">(optional, for reproducibility)</span>
                </span>
                <input
                  type="number"
                  value={seed ?? ''}
                  onChange={(e) => setSeed(e.target.value === '' ? null : parseInt(e.target.value))}
                  placeholder="Random if empty"
                  min="0"
                  className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none placeholder:text-slate-600"
                />
                <p className="text-xs text-notion-text-tertiary">
                  Set a fixed seed to reproduce exact training results. Leave empty for random initialization.
                </p>
              </label>

              {tab !== 'Simulation' && (
                <label className="block space-y-2">
                  <span className="text-notion-text">Performance Mode</span>
                  <div className="flex items-start gap-3 rounded-xl border border-notion-border bg-notion-light-gray px-3 py-2">
                    <input
                      type="checkbox"
                      checked={runWithoutSimulation}
                      onChange={(e) => setRunWithoutSimulation(e.target.checked)}
                      className="mt-0.5 rounded border-slate-600 bg-slate-900/80 text-notion-blue focus:ring-cyan-400"
                    />
                    <div>
                      <p className="text-sm text-notion-text">Run without simulation (faster, no replay)</p>
                      <p className="text-xs text-notion-text-tertiary">
                        Skips simulation frame streaming and trajectory recording for maximum speed.
                      </p>
                    </div>
                  </div>
                </label>
              )}

              {tab !== 'Simulation' && tab !== 'Evaluation' && (
                <div className={`rounded-xl border border-notion-border bg-slate-50 p-4 transition-all ${(tab === 'Residual' || (tab === 'Fine-Tuning' && safetyConstraintEnabled)) ? 'ring-1 ring-orange-500/50 bg-orange-50' : ''
                  }`}>
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-semibold text-notion-text flex items-center gap-2">
                      🛡️ Safety Constraints
                    </span>

                    {/* Toggle switch for Training AND Fine-Tuning. Forced ON for Residual only */}
                    {(tab === 'Training' || tab === 'Fine-Tuning') ? (
                      <label className="relative inline-flex cursor-pointer items-center">
                        <input
                          type="checkbox"
                          className="peer sr-only"
                          checked={safetyConstraintEnabled}
                          onChange={(e) => setSafetyConstraintEnabled(e.target.checked)}
                        />
                        <div className="peer h-5 w-9 rounded-full bg-slate-700 after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:bg-white after:transition-all after:content-[''] peer-checked:bg-cyan-500 peer-checked:after:translate-x-full peer-focus:outline-none"></div>
                      </label>
                    ) : (tab === 'Residual') ? (
                      <span className="text-xs font-semibold text-orange-600 bg-orange-100 px-2 py-0.5 rounded border border-orange-200">
                        ALWAYS ON
                      </span>
                    ) : null}
                  </div>

                  {/* Only show config if enabled (or if forced enabled) */}
                  {(safetyConstraintEnabled || tab === 'Residual') && (
                    <div className="space-y-3 animate-in fade-in slide-in-from-top-2 duration-200">
                      <p className="text-xs text-notion-text-secondary">
                        Lagrangian relaxation to satisfy cost constraints.
                      </p>

                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <label className="space-y-1">
                          <span className="text-notion-text-secondary text-xs">Risk Budget (ε)</span>
                          <input
                            type="number"
                            step="0.01"
                            value={riskBudget}
                            onChange={(e) => setRiskBudget(parseFloat(e.target.value))}
                            className="w-full rounded-lg border border-slate-700 bg-notion-hover px-2 py-1.5 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none text-sm"
                          />
                        </label>
                        <label className="space-y-1">
                          <span className="text-notion-text-secondary text-xs">Initial Lambda (λ₀)</span>
                          <input
                            type="number"
                            step="0.01"
                            value={initialLambda}
                            onChange={(e) => setInitialLambda(parseFloat(e.target.value))}
                            className="w-full rounded-lg border border-slate-700 bg-notion-hover px-2 py-1.5 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none text-sm"
                          />
                        </label>
                        <label className="space-y-1">
                          <span className="text-notion-text-secondary text-xs">Lambda Learning Rate</span>
                          <input
                            type="number"
                            step="0.001"
                            value={lambdaLearningRate}
                            onChange={(e) => setLambdaLearningRate(parseFloat(e.target.value))}
                            className="w-full rounded-lg border border-slate-700 bg-notion-hover px-2 py-1.5 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none text-sm"
                          />
                        </label>
                        <label className="space-y-1">
                          <span className="text-notion-text-secondary text-xs">Update Freq (eps)</span>
                          <input
                            type="number"
                            step="1"
                            value={updateFrequency}
                            onChange={(e) => setUpdateFrequency(parseInt(e.target.value))}
                            className="w-full rounded-lg border border-slate-700 bg-notion-hover px-2 py-1.5 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none text-sm"
                          />
                        </label>
                      </div>

                      {/* Lambda Warmup Schedule Section */}
                      <div className="space-y-2 pt-3 border-t border-notion-border/50">
                        <div className="flex items-center gap-2">
                          <span className="text-notion-text text-xs font-semibold">🔥 Lambda Warmup Schedule</span>
                          <span className="text-notion-text-tertiary text-[10px]">(Optional)</span>
                        </div>
                        <div className="grid grid-cols-3 gap-2">
                          <label className="space-y-1">
                            <span className="text-notion-text-secondary text-xs">Target λ</span>
                            <input
                              type="number"
                              step="1"
                              placeholder="None"
                              value={targetLambda ?? ''}
                              onChange={(e) => setTargetLambda(e.target.value ? parseFloat(e.target.value) : null)}
                              className="w-full rounded-lg border border-slate-700 bg-notion-hover px-2 py-1.5 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none text-sm placeholder:text-slate-600"
                            />
                          </label>
                          <label className="space-y-1">
                            <span className="text-notion-text-secondary text-xs">Warmup Episodes</span>
                            <input
                              type="number"
                              step="1"
                              value={warmupEpisodes}
                              onChange={(e) => setWarmupEpisodes(parseInt(e.target.value) || 0)}
                              className="w-full rounded-lg border border-slate-700 bg-notion-hover px-2 py-1.5 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none text-sm"
                            />
                          </label>
                          <label className="space-y-1">
                            <span className="text-notion-text-secondary text-xs">Schedule</span>
                            <select
                              value={warmupSchedule}
                              onChange={(e) => setWarmupSchedule(e.target.value)}
                              className="w-full rounded-lg border border-slate-700 bg-notion-hover px-2 py-1.5 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none text-sm"
                            >
                              <option value="exponential">Exponential</option>
                              <option value="linear">Linear</option>
                            </select>
                          </label>
                        </div>
                        {targetLambda && warmupEpisodes > 0 && (
                          <div className="text-[10px] text-notion-text-tertiary bg-notion-light-gray rounded px-2 py-1">
                            λ will gradually increase from 0 to {targetLambda} over {warmupEpisodes} episodes using {warmupSchedule} schedule
                          </div>
                        )}
                      </div>

                      <div className="space-y-2 pt-2 border-t border-notion-border/50">
                        <label className="space-y-1 block">
                          <span className="text-notion-text-secondary text-xs">Cost Signal Preset</span>
                          <select
                            value={costSignalPreset}
                            onChange={(e) => setCostSignalPreset(e.target.value)}
                            className="w-full rounded-lg border border-slate-700 bg-notion-hover px-2 py-1.5 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none text-sm"
                          >
                            <option value="balanced">Balanced (Default)</option>
                            <option value="strict_safety">Strict Safety (High Collision Pen)</option>
                            <option value="near_miss_focused">Near-Miss Focused</option>
                            <option value="collision_only">Collision Only</option>
                          </select>
                        </label>

                        <label className="space-y-1 block">
                          <span className="text-notion-text-secondary text-xs">Near-Miss Threshold (distance)</span>
                          <input
                            type="number"
                            step="0.1"
                            value={nearMissThreshold}
                            onChange={(e) => setNearMissThreshold(parseFloat(e.target.value))}
                            className="w-full rounded-lg border border-slate-700 bg-notion-hover px-2 py-1.5 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none text-sm"
                          />
                        </label>

                        <div className="space-y-1.5">
                          <span className="text-notion-text-secondary text-xs block">Cost Weights</span>
                          <div className="grid grid-cols-3 gap-2">
                            <label className="space-y-1">
                              <span className="text-notion-text-tertiary text-[10px] block">Collision</span>
                              <input
                                type="number"
                                step="0.1"
                                value={collisionWeight}
                                onChange={(e) => setCollisionWeight(parseFloat(e.target.value))}
                                className="w-full rounded border border-slate-700 bg-notion-hover px-2 py-1 text-xs text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                              />
                            </label>
                            <label className="space-y-1">
                              <span className="text-notion-text-tertiary text-[10px] block">Near-Miss</span>
                              <input
                                type="number"
                                step="0.1"
                                value={nearMissWeight}
                                onChange={(e) => setNearMissWeight(parseFloat(e.target.value))}
                                className="w-full rounded border border-slate-700 bg-notion-hover px-2 py-1 text-xs text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                              />
                            </label>
                            <label className="space-y-1">
                              <span className="text-notion-text-tertiary text-[10px] block">Danger</span>
                              <input
                                type="number"
                                step="0.1"
                                value={dangerZoneWeight}
                                onChange={(e) => setDangerZoneWeight(parseFloat(e.target.value))}
                                className="w-full rounded border border-slate-700 bg-notion-hover px-2 py-1 text-xs text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                              />
                            </label>
                          </div>
                        </div>

                        <div className="pt-2 border-t border-notion-border/50">
                          <label className="flex items-center gap-2 cursor-pointer text-sm">
                            <input
                              type="checkbox"
                              checked={ignoreWalls}
                              onChange={(e) => setIgnoreWalls(e.target.checked)}
                              className="rounded border-slate-700 bg-notion-hover text-cyan-500 focus:ring-cyan-400 focus:ring-offset-slate-900"
                            />
                            <span className="text-notion-text">Ignore Walls for Safety</span>
                            <span className="text-notion-text-tertiary text-xs">(crashes + proximity ignored)</span>
                          </label>
                        </div>

                        <div className="pt-2 border-t border-notion-border/50 space-y-2">
                          <span className="text-notion-text-secondary text-xs block">Wall Proximity Costs (Training Gradient)</span>
                          <p className="text-[10px] text-notion-text-tertiary">These costs provide a smooth gradient for RL training only when ignore_walls=false</p>

                          <div className="grid grid-cols-3 gap-2">
                            <label className="space-y-1">
                              <span className="text-notion-text-tertiary text-[10px] block">Wall Near-Miss</span>
                              <input
                                type="number"
                                step="0.001"
                                value={wallNearMissWeight}
                                onChange={(e) => setWallNearMissWeight(parseFloat(e.target.value))}
                                className="w-full rounded border border-slate-700 bg-notion-hover px-2 py-1 text-xs text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                              />
                            </label>
                            <label className="space-y-1">
                              <span className="text-notion-text-tertiary text-[10px] block">Wall Danger</span>
                              <input
                                type="number"
                                step="0.001"
                                value={wallDangerZoneWeight}
                                onChange={(e) => setWallDangerZoneWeight(parseFloat(e.target.value))}
                                className="w-full rounded border border-slate-700 bg-notion-hover px-2 py-1 text-xs text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                              />
                            </label>
                            <label className="space-y-1">
                              <span className="text-notion-text-tertiary text-[10px] block">Wall Threshold (m)</span>
                              <input
                                type="number"
                                step="0.1"
                                value={wallNearMissThreshold}
                                onChange={(e) => setWallNearMissThreshold(parseFloat(e.target.value))}
                                className="w-full rounded border border-slate-700 bg-notion-hover px-2 py-1 text-xs text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                              />
                            </label>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="col-span-2 flex justify-end gap-3 pt-4">
              <button
                type="button"
                onClick={onClose}
                className="rounded-xl px-4 py-2 text-sm font-semibold text-notion-text-secondary hover:text-notion-text"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={loading || environments.length === 0 || !environmentId}
                className="rounded-xl bg-cyan-500 px-6 py-2 text-sm font-semibold text-white shadow-lg shadow-notion-blue/20 hover:bg-cyan-400 disabled:opacity-50"
                title={!environmentId && environments.length > 0 ? 'Please select an environment' : ''}
              >
                {loading ? 'Creating...' : environments.length === 0 ? 'Loading...' : !environmentId ? 'Select Environment' : 'Create Experiment'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
