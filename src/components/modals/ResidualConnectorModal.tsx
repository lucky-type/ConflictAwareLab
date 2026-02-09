import { useEffect, useState, FormEvent } from 'react';
import { X, Lock } from 'lucide-react';
import { Card } from '../common/Card';
import { ResidualConnector, Agent, api } from '../../services/api';

interface Props {
  open: boolean;
  onClose: () => void;
  onSave: (data: {
    name: string;
    parent_agent_id: number;
    algorithm: string;
    net_arch: number[];
    parameters: Record<string, any>;
    k_factor: number;
    adaptive_k: boolean;
    enable_k_conf: boolean;
    enable_k_risk: boolean;
  }) => void;
  connector?: ResidualConnector | null;
}

const defaultParameters: Record<string, Record<string, any>> = {
  PPO: {
    learning_rate: 0.0003,
    n_steps: 2048,
    batch_size: 64,
    n_epochs: 10,
    gamma: 0.99,
    gae_lambda: 0.95,
    clip_range: 0.2,
    ent_coef: 0.0,
  },
  A2C: {
    learning_rate: 0.0007,
    n_steps: 5,
    gamma: 0.99,
    gae_lambda: 1.0,
    ent_coef: 0.0,
    vf_coef: 0.5,
  },
};

const parameterDefinitions: Record<string, Array<{ key: string; label: string; type: 'number' | 'text' }>> = {
  PPO: [
    { key: 'learning_rate', label: 'Learning Rate', type: 'number' },
    { key: 'n_steps', label: 'Steps per Rollout', type: 'number' },
    { key: 'batch_size', label: 'Batch Size', type: 'number' },
    { key: 'n_epochs', label: 'Epochs', type: 'number' },
    { key: 'gamma', label: 'Discount Factor (γ)', type: 'number' },
    { key: 'gae_lambda', label: 'GAE Lambda (λ)', type: 'number' },
    { key: 'clip_range', label: 'Clip Range', type: 'number' },
    { key: 'ent_coef', label: 'Entropy Coefficient', type: 'number' },
  ],
  A2C: [
    { key: 'learning_rate', label: 'Learning Rate', type: 'number' },
    { key: 'n_steps', label: 'Steps per Update', type: 'number' },
    { key: 'gamma', label: 'Discount Factor (γ)', type: 'number' },
    { key: 'gae_lambda', label: 'GAE Lambda (λ)', type: 'number' },
    { key: 'ent_coef', label: 'Entropy Coefficient', type: 'number' },
    { key: 'vf_coef', label: 'Value Function Coefficient', type: 'number' },
  ],
};

export default function ResidualConnectorModal({ open, onClose, onSave, connector }: Props) {
  const [name, setName] = useState('');
  const [parentAgentId, setParentAgentId] = useState<number | ''>('');
  const [algorithm, setAlgorithm] = useState<string>('PPO');
  const [netArch, setNetArch] = useState<string>('64, 64');
  const [kFactor, setKFactor] = useState<number>(0.15);
  const [adaptiveK, setAdaptiveK] = useState<boolean>(false);
  const [enableKConf, setEnableKConf] = useState<boolean>(true);
  const [enableKRisk, setEnableKRisk] = useState<boolean>(true);
  const [parameters, setParameters] = useState<Record<string, any>>({});
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [loadingAgents, setLoadingAgents] = useState(false);

  // Load available agents
  useEffect(() => {
    if (open) {
      loadAgents();
    }
  }, [open]);

  // Initialize form when modal opens
  useEffect(() => {
    if (!open) return;

    if (connector) {
      setName(connector.name);
      setParentAgentId(connector.parent_agent_id);
      setAlgorithm(connector.algorithm);
      setNetArch(connector.net_arch.join(', '));
      setKFactor((connector as any).k_factor ?? 0.15);
      setAdaptiveK((connector as any).adaptive_k ?? false);
      setEnableKConf((connector as any).enable_k_conf ?? true);
      setEnableKRisk((connector as any).enable_k_risk ?? true);
      setParameters(connector.parameters);

      // Load parent agent details
      if (connector.parent_agent) {
        setSelectedAgent(connector.parent_agent);
      }
    } else {
      setName('');
      setParentAgentId('');
      setAlgorithm('PPO');
      setNetArch('64, 64');
      setKFactor(0.15);
      setAdaptiveK(false);
      setEnableKConf(true);
      setEnableKRisk(true);
      setParameters({ ...defaultParameters.PPO });
      setSelectedAgent(null);
    }
  }, [open, connector]);

  // Update parameters when algorithm changes
  useEffect(() => {
    if (!connector && algorithm) {
      setParameters({ ...defaultParameters[algorithm] });
    }
  }, [algorithm, connector]);

  // Update selected agent when parent agent ID changes
  useEffect(() => {
    if (parentAgentId && agents.length > 0) {
      const agent = agents.find(a => a.id === parentAgentId);
      setSelectedAgent(agent || null);
    }
  }, [parentAgentId, agents]);

  const loadAgents = async () => {
    try {
      setLoadingAgents(true);
      const agentList = await api.getAgents();
      setAgents(agentList);
    } catch (err) {
      console.error('Failed to load agents:', err);
    } finally {
      setLoadingAgents(false);
    }
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();

    // Parse network architecture
    const archArray = netArch.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));

    onSave({
      name,
      parent_agent_id: typeof parentAgentId === 'number' ? parentAgentId : parseInt(String(parentAgentId)),
      algorithm,
      net_arch: archArray,
      parameters,
      k_factor: kFactor,
      adaptive_k: adaptiveK,
      enable_k_conf: enableKConf,
      enable_k_risk: enableKRisk,
    });
  };

  const currentParams = parameterDefinitions[algorithm] || [];

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm"
    >
      <div className="w-full max-w-2xl overflow-y-auto max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
        <Card className="p-6">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs uppercase tracking-wide text-purple-600">
                {connector ? 'Edit Residual Connector' : 'New Residual Connector'}
              </p>
              <h3 className="text-lg font-semibold text-notion-text">{name || 'Unnamed Connector'}</h3>
            </div>
            <button onClick={onClose} className="text-notion-text-secondary hover:text-notion-text">
              <X size={20} />
            </button>
          </div>

          <form onSubmit={handleSubmit} className="mt-6 space-y-5">
            {/* Step 1: Identity & Inheritance */}
            <div className="space-y-3 rounded-xl border border-purple-200 bg-purple-50 p-4">
              <div>
                <p className="text-sm font-semibold text-notion-text">Step 1: Identity & Inheritance</p>
                <p className="text-xs text-notion-text-secondary">Name the connector and select the parent agent</p>
              </div>

              <label className="block space-y-2 text-sm">
                <span className="text-notion-text">Connector Name</span>
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-purple-400"
                  placeholder="e.g. Safe_Refiner_v1"
                  required
                />
              </label>

              <label className="block space-y-2 text-sm">
                <span className="text-notion-text">Parent Agent</span>
                <select
                  value={parentAgentId}
                  onChange={(e) => setParentAgentId(e.target.value === '' ? '' : parseInt(e.target.value))}
                  className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-purple-400"
                  required
                  disabled={!!connector || loadingAgents}
                >
                  <option value="">Select a parent agent...</option>
                  {agents.map((agent) => (
                    <option key={agent.id} value={agent.id}>
                      {agent.name} ({agent.type})
                    </option>
                  ))}
                </select>
                {connector && (
                  <p className="text-xs text-amber-400">Parent agent cannot be changed after creation</p>
                )}
              </label>
            </div>

            {/* Step 2: Auto-Fill & Lock Physical Specs */}
            {selectedAgent && (
              <div className="space-y-3 rounded-xl border border-notion-border bg-notion-light-gray p-4">
                <div>
                  <p className="text-sm font-semibold text-notion-text">Step 2: Inherited Physical Specs (Read-Only)</p>
                  <p className="text-xs text-notion-text-secondary">These properties are locked to match the parent agent</p>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <label className="space-y-1.5 text-sm">
                    <span className="flex items-center gap-2 text-notion-text">
                      <Lock size={14} className="text-notion-text-tertiary" />
                      Diameter (m)
                    </span>
                    <input
                      type="number"
                      value={selectedAgent.agent_diameter}
                      className="w-full rounded-xl border border-notion-border bg-notion-light-gray px-3 py-2 text-notion-text-secondary cursor-not-allowed"
                      disabled
                      readOnly
                    />
                  </label>

                  <label className="space-y-1.5 text-sm">
                    <span className="flex items-center gap-2 text-notion-text">
                      <Lock size={14} className="text-notion-text-tertiary" />
                      Max Speed (m/s)
                    </span>
                    <input
                      type="number"
                      value={selectedAgent.agent_max_speed}
                      className="w-full rounded-xl border border-notion-border bg-notion-light-gray px-3 py-2 text-notion-text-secondary cursor-not-allowed"
                      disabled
                      readOnly
                    />
                  </label>

                  <label className="space-y-1.5 text-sm">
                    <span className="flex items-center gap-2 text-notion-text">
                      <Lock size={14} className="text-notion-text-tertiary" />
                      Lidar Range (m)
                    </span>
                    <input
                      type="number"
                      value={selectedAgent.lidar_range}
                      className="w-full rounded-xl border border-notion-border bg-notion-light-gray px-3 py-2 text-notion-text-secondary cursor-not-allowed"
                      disabled
                      readOnly
                    />
                  </label>

                  <label className="space-y-1.5 text-sm">
                    <span className="flex items-center gap-2 text-notion-text">
                      <Lock size={14} className="text-notion-text-tertiary" />
                      Lidar Rays
                    </span>
                    <input
                      type="number"
                      value={selectedAgent.lidar_rays}
                      className="w-full rounded-xl border border-notion-border bg-notion-light-gray px-3 py-2 text-notion-text-secondary cursor-not-allowed"
                      disabled
                      readOnly
                    />
                  </label>
                </div>
              </div>
            )}

            {/* Step 3: Residual Policy Configuration */}
            <div className="space-y-3 rounded-xl border border-purple-200 bg-purple-50 p-4">
              <div>
                <p className="text-sm font-semibold text-notion-text">Step 3: Residual Policy Configuration</p>
                <p className="text-xs text-notion-text-secondary">Define the residual learner algorithm and network</p>
              </div>

              <label className="block space-y-2 text-sm">
                <span className="text-notion-text">Algorithm</span>
                <select
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value)}
                  className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-purple-400"
                  disabled={!!connector}
                >
                  <option value="PPO">PPO - Proximal Policy Optimization</option>
                  <option value="A2C">A2C - Advantage Actor-Critic</option>
                </select>
                {connector && (
                  <p className="text-xs text-amber-400">Algorithm cannot be changed after creation</p>
                )}
              </label>

              <label className="block space-y-2 text-sm">
                <span className="text-notion-text">Network Architecture</span>
                <input
                  value={netArch}
                  onChange={(e) => setNetArch(e.target.value)}
                  className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-purple-400"
                  placeholder="e.g., 64, 64"
                  required
                />
                <p className="text-xs text-notion-text-tertiary">
                  Comma-separated list of hidden layer sizes (e.g., 64, 64 or 128, 128, 64)
                </p>
              </label>

              <label className="block space-y-2 text-sm">
                <span className="text-notion-text">K-Factor (Scaling)</span>
                <input
                  type="number"
                  step="0.01"
                  min="0.01"
                  max="1.0"
                  value={kFactor}
                  onChange={(e) => setKFactor(parseFloat(e.target.value))}
                  className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-purple-400"
                  required
                />
                <p className="text-xs text-notion-text-tertiary">
                  Scaling factor for residual corrections (0.01-1.0, typically 0.05-0.5). Higher values = more aggressive corrections.
                </p>
              </label>

              {/* Adaptive K-Factor Toggle */}
              <label className="flex items-center gap-3 rounded-xl border border-purple-200 bg-purple-100 p-3 cursor-pointer hover:bg-purple-50 transition">
                <input
                  type="checkbox"
                  checked={adaptiveK}
                  onChange={(e) => setAdaptiveK(e.target.checked)}
                  className="w-5 h-5 rounded border-purple-300 bg-white text-purple-600 focus:ring-purple-400"
                />
                <div>
                  <span className="text-sm font-medium text-notion-text">Adaptive K-Factor (CARS)</span>
                  <p className="text-xs text-notion-text-secondary">
                    CARS: Conflict-Aware Residual Scaling. Smoothly reduces K (min 70% of base) when residual strongly opposes base action.
                    Considers risk (λ) and conflict severity.
                  </p>
                </div>
              </label>

              {/* CARS Ablation Control (only show when CARS is enabled) */}
              {adaptiveK && (
                <div className="space-y-2 rounded-xl border border-amber-200 bg-amber-50 p-4">
                  <div>
                    <p className="text-sm font-semibold text-amber-700">CARS Component Control (Ablation)</p>
                    <p className="text-xs text-notion-text-secondary">
                      Enable/disable individual CARS components for ablation studies. All enabled by default.
                    </p>
                  </div>

                  <label className="flex items-center gap-3 rounded-lg border border-slate-700/40 bg-notion-light-gray p-2.5 cursor-pointer hover:bg-notion-light-gray transition">
                    <input
                      type="checkbox"
                      checked={enableKConf}
                      onChange={(e) => setEnableKConf(e.target.checked)}
                      className="w-4 h-4 rounded border-slate-600 bg-slate-900 text-amber-500 focus:ring-amber-400"
                    />
                    <div className="flex-1">
                      <span className="text-sm font-medium text-notion-text">K_conf (Conflict-Aware)</span>
                      <p className="text-xs text-notion-text-secondary">Reduces K when residual opposes base action (cosine similarity)</p>
                    </div>
                  </label>

                  <label className="flex items-center gap-3 rounded-lg border border-slate-700/40 bg-notion-light-gray p-2.5 cursor-pointer hover:bg-notion-light-gray transition">
                    <input
                      type="checkbox"
                      checked={enableKRisk}
                      onChange={(e) => setEnableKRisk(e.target.checked)}
                      className="w-4 h-4 rounded border-slate-600 bg-slate-900 text-amber-500 focus:ring-amber-400"
                    />
                    <div className="flex-1">
                      <span className="text-sm font-medium text-notion-text">K_risk (Risk-Aware)</span>
                      <p className="text-xs text-notion-text-secondary">Reduces K when Lagrange multiplier λ is high</p>
                    </div>
                  </label>
                </div>
              )}
            </div>

            {/* Hyperparameters */}
            <div className="space-y-3 rounded-xl border border-notion-border bg-notion-light-gray p-4">
              <div>
                <p className="text-sm font-semibold text-notion-text">Residual Algorithm Hyperparameters</p>
                <p className="text-xs text-notion-text-secondary">Configure the parameters for the residual training process</p>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                {currentParams.map((param) => (
                  <label key={param.key} className="space-y-1.5 text-sm">
                    <span className="text-notion-text">{param.label}</span>
                    <input
                      type={param.type === 'number' ? 'number' : 'text'}
                      step={param.type === 'number' ? 'any' : undefined}
                      value={parameters[param.key] ?? ''}
                      onChange={(e) => {
                        const value = param.type === 'number' ? parseFloat(e.target.value) : e.target.value;
                        setParameters((prev) => ({ ...prev, [param.key]: value }));
                      }}
                      className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-purple-400"
                    />
                  </label>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-end gap-3">
              <button
                type="button"
                onClick={onClose}
                className="rounded-xl border border-notion-border px-4 py-2 text-sm text-notion-text hover:border-purple-400/50"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="rounded-xl bg-notion-blue px-4 py-2 text-sm font-semibold text-white hover:opacity-90"
              >
                Save
              </button>
            </div>
          </form>
        </Card>
      </div>
    </div>
  );
}
