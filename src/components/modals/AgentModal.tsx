import { useEffect, useState, FormEvent } from 'react';
import { X } from 'lucide-react';
import { Card } from '../common/Card';
import { Agent } from '../../services/api';

interface Props {
  open: boolean;
  onClose: () => void;
  onSave: (data: {
    name: string;
    type: string;
    agent_diameter: number;
    agent_max_speed: number;
    lidar_range: number;
    lidar_rays: number;
    kinematic_type: string;
    parameters: Record<string, any>
  }) => void;
  agent?: Agent | null;
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
  SAC: {
    learning_rate: 0.0003,
    buffer_size: 1000000,
    batch_size: 256,
    tau: 0.005,
    gamma: 0.99,
    ent_coef: 'auto',
    target_entropy: 'auto',
  },
  TD3: {
    learning_rate: 0.001,
    buffer_size: 1000000,
    batch_size: 256,
    tau: 0.005,
    gamma: 0.99,
    policy_delay: 2,
    target_noise: 0.2,
  },
  A2C: {
    learning_rate: 0.0007,
    n_steps: 5,
    gamma: 0.99,
    gae_lambda: 1.0,
    ent_coef: 0.0,
    vf_coef: 0.5,
  },
  DDPG: {
    learning_rate: 0.001,
    buffer_size: 1000000,
    batch_size: 256,
    tau: 0.005,
    gamma: 0.99,
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
  SAC: [
    { key: 'learning_rate', label: 'Learning Rate', type: 'number' },
    { key: 'buffer_size', label: 'Replay Buffer Size', type: 'number' },
    { key: 'batch_size', label: 'Batch Size', type: 'number' },
    { key: 'tau', label: 'Polyak Tau (τ)', type: 'number' },
    { key: 'gamma', label: 'Discount Factor (γ)', type: 'number' },
    { key: 'ent_coef', label: 'Entropy Coefficient', type: 'text' },
    { key: 'target_entropy', label: 'Target Entropy', type: 'text' },
  ],
  TD3: [
    { key: 'learning_rate', label: 'Learning Rate', type: 'number' },
    { key: 'buffer_size', label: 'Replay Buffer Size', type: 'number' },
    { key: 'batch_size', label: 'Batch Size', type: 'number' },
    { key: 'tau', label: 'Polyak Tau (τ)', type: 'number' },
    { key: 'gamma', label: 'Discount Factor (γ)', type: 'number' },
    { key: 'policy_delay', label: 'Policy Delay', type: 'number' },
    { key: 'target_noise', label: 'Target Noise', type: 'number' },
  ],
  A2C: [
    { key: 'learning_rate', label: 'Learning Rate', type: 'number' },
    { key: 'n_steps', label: 'Steps per Update', type: 'number' },
    { key: 'gamma', label: 'Discount Factor (γ)', type: 'number' },
    { key: 'gae_lambda', label: 'GAE Lambda (λ)', type: 'number' },
    { key: 'ent_coef', label: 'Entropy Coefficient', type: 'number' },
    { key: 'vf_coef', label: 'Value Function Coefficient', type: 'number' },
  ],
  DDPG: [
    { key: 'learning_rate', label: 'Learning Rate', type: 'number' },
    { key: 'buffer_size', label: 'Replay Buffer Size', type: 'number' },
    { key: 'batch_size', label: 'Batch Size', type: 'number' },
    { key: 'tau', label: 'Polyak Tau (τ)', type: 'number' },
    { key: 'gamma', label: 'Discount Factor (γ)', type: 'number' },
  ],
};

export default function AgentModal({ open, onClose, onSave, agent }: Props) {
  const [name, setName] = useState('');
  const [type, setType] = useState<string>('PPO');
  const [agentDiameter, setAgentDiameter] = useState<number | ''>(1.2);
  const [agentMaxSpeed, setAgentMaxSpeed] = useState<number | ''>(6);
  const [lidarRange, setLidarRange] = useState<number | ''>(20);
  const [lidarRays, setLidarRays] = useState<number | ''>(36);
  const [kinematicType, setKinematicType] = useState<string>('holonomic');
  const [parameters, setParameters] = useState<Record<string, any>>({});

  useEffect(() => {
    if (!open) return;

    if (agent) {
      setName(agent.name);
      setType(agent.type);
      setAgentDiameter(agent.agent_diameter);
      setAgentMaxSpeed(agent.agent_max_speed);
      setLidarRange(agent.lidar_range);
      setLidarRays(agent.lidar_rays);
      setKinematicType(agent.kinematic_type || 'holonomic');
      setParameters(agent.parameters);
    } else {
      setName('');
      setType('PPO');
      setAgentDiameter(1.2);
      setAgentMaxSpeed(6);
      setLidarRange(20);
      setLidarRays(36);
      setKinematicType('holonomic');
      setParameters({ ...defaultParameters.PPO });
    }
  }, [open, agent]);

  // Update parameters when algorithm type changes
  useEffect(() => {
    if (!agent) {
      setParameters({ ...defaultParameters[type] });
    }
  }, [type, agent]);

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    onSave({
      name,
      type,
      agent_diameter: typeof agentDiameter === 'number' ? agentDiameter : parseFloat(String(agentDiameter)) || 1.2,
      agent_max_speed: typeof agentMaxSpeed === 'number' ? agentMaxSpeed : parseFloat(String(agentMaxSpeed)) || 6,
      lidar_range: typeof lidarRange === 'number' ? lidarRange : parseFloat(String(lidarRange)) || 20,
      lidar_rays: typeof lidarRays === 'number' ? lidarRays : parseInt(String(lidarRays)) || 36,
      kinematic_type: kinematicType,
      parameters
    });
  };

  const currentParams = parameterDefinitions[type] || [];

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm"
    >
      <div className="w-full max-w-2xl overflow-y-auto max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
        <Card className="p-6">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs uppercase tracking-wide text-notion-blue">
                {agent ? 'Edit Agent' : 'New Agent'}
              </p>
              <h3 className="text-lg font-semibold text-notion-text">{name || 'Unnamed Agent'}</h3>
            </div>
            <button onClick={onClose} className="text-notion-text-secondary hover:text-notion-text">
              <X size={20} />
            </button>
          </div>

          <form onSubmit={handleSubmit} className="mt-6 space-y-5">
            <label className="block space-y-2 text-sm">
              <span className="text-notion-text">Agent Name</span>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                placeholder="e.g. PPO_Drone_v1"
                required
              />
            </label>

            <label className="block space-y-2 text-sm">
              <span className="text-notion-text">Algorithm Type</span>
              <select
                value={type}
                onChange={(e) => setType(e.target.value)}
                className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                disabled={!!agent}
              >
                <option value="PPO">PPO - Proximal Policy Optimization</option>
                <option value="SAC">SAC - Soft Actor-Critic</option>
                <option value="TD3">TD3 - Twin Delayed DDPG</option>
                <option value="A2C">A2C - Advantage Actor-Critic</option>
                <option value="DDPG">DDPG - Deep Deterministic Policy Gradient</option>
              </select>
              {agent && (
                <p className="text-xs text-amber-400">Algorithm type cannot be changed after creation</p>
              )}
            </label>

            <label className="block space-y-2 text-sm">
              <span className="text-notion-text">Kinematic Model</span>
              <select
                value={kinematicType}
                onChange={(e) => setKinematicType(e.target.value)}
                className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                disabled={!!agent}
              >
                <option value="holonomic">Holonomic - Direct velocity control (vx, vy, vz)</option>
                <option value="semi-holonomic">Semi-holonomic - Forward + yaw control (more realistic)</option>
              </select>
              {agent && (
                <p className="text-xs text-amber-400">Kinematic type cannot be changed after creation</p>
              )}
              <p className="text-xs text-notion-text-tertiary">
                {kinematicType === 'holonomic'
                  ? '✓ Most stable for RL training, fastest convergence, can move in any direction'
                  : '✓ Closer to real UAV dynamics, better for robustness and curriculum learning'}
              </p>
            </label>

            {/* Agent Physical Properties */}
            <div className="space-y-3 rounded-xl border border-notion-border bg-notion-light-gray p-4">
              <div>
                <p className="text-sm font-semibold text-notion-text">Agent Physical Configuration</p>
                <p className="text-xs text-notion-text-secondary">Define the physical properties of the drone agent</p>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <label className="space-y-1.5 text-sm">
                  <span className="text-notion-text">Diameter (m)</span>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="10"
                    value={agentDiameter}
                    onChange={(e) => setAgentDiameter(e.target.value === '' ? '' : parseFloat(e.target.value))}
                    className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                    required
                  />
                </label>

                <label className="space-y-1.5 text-sm">
                  <span className="text-notion-text">Max Speed (m/s)</span>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="100"
                    value={agentMaxSpeed}
                    onChange={(e) => setAgentMaxSpeed(e.target.value === '' ? '' : parseFloat(e.target.value))}
                    className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                    required
                  />
                </label>

                <label className="space-y-1.5 text-sm">
                  <span className="text-notion-text">Lidar Range (m)</span>
                  <input
                    type="number"
                    step="1"
                    min="1"
                    max="200"
                    value={lidarRange}
                    onChange={(e) => setLidarRange(e.target.value === '' ? '' : parseFloat(e.target.value))}
                    className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                    required
                  />
                </label>

                <label className="space-y-1.5 text-sm">
                  <span className="text-notion-text">Lidar Rays</span>
                  <input
                    type="number"
                    step="1"
                    min="4"
                    max="360"
                    value={lidarRays}
                    onChange={(e) => setLidarRays(e.target.value === '' ? '' : parseInt(e.target.value))}
                    className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                    required
                  />
                </label>
              </div>
            </div>

            {/* Algorithm Hyperparameters */}
            <div className="space-y-3 rounded-xl border border-notion-border bg-notion-light-gray p-4">
              <div>
                <p className="text-sm font-semibold text-notion-text">Algorithm Hyperparameters</p>
                <p className="text-xs text-notion-text-secondary">Configure the algorithm parameters (from Stable-Baselines3)</p>
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
                      className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                    />
                  </label>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-end gap-3">
              <button
                type="button"
                onClick={onClose}
                className="rounded-xl border border-notion-border px-4 py-2 text-sm text-notion-text hover:bg-notion-hover"
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
