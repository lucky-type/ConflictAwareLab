import { useEffect, useState } from 'react';
import { PenSquare, Plus, Trash2 } from 'lucide-react';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import AgentModal from '../components/modals/AgentModal';
import { api, type Agent as APIAgent } from '../services/api';
import { useTopBar } from '../context/TopBarContext';

const formatValue = (value: number | string) => {
  if (typeof value === 'number') {
    if (Math.abs(value) >= 1000 || Math.abs(value) < 0.01) {
      return value.toExponential(2);
    }
    return Number.isInteger(value) ? value.toString() : value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
  }
  return value;
};

// Agent type definitions with parameters
const agentTemplates: Record<string, { description: string; parameters: Array<{ key: string; label: string }> }> = {
  PPO: {
    description: 'Proximal Policy Optimization - On-policy actor-critic algorithm with clipped objective',
    parameters: [
      { key: 'learning_rate', label: 'Learning Rate' },
      { key: 'n_steps', label: 'Steps per Update' },
      { key: 'batch_size', label: 'Batch Size' },
      { key: 'n_epochs', label: 'Epochs' },
      { key: 'gamma', label: 'Discount Factor' },
      { key: 'gae_lambda', label: 'GAE Lambda' },
      { key: 'clip_range', label: 'Clip Range' },
      { key: 'ent_coef', label: 'Entropy Coefficient' },
    ],
  },
  SAC: {
    description: 'Soft Actor-Critic - Off-policy algorithm that maximizes entropy-regularized expected return',
    parameters: [
      { key: 'learning_rate', label: 'Learning Rate' },
      { key: 'buffer_size', label: 'Buffer Size' },
      { key: 'batch_size', label: 'Batch Size' },
      { key: 'tau', label: 'Tau (soft update)' },
      { key: 'gamma', label: 'Discount Factor' },
      { key: 'ent_coef', label: 'Entropy Coefficient' },
      { key: 'target_entropy', label: 'Target Entropy' },
    ],
  },
  TD3: {
    description: 'Twin Delayed DDPG - Off-policy algorithm with double Q-learning and delayed policy updates',
    parameters: [
      { key: 'learning_rate', label: 'Learning Rate' },
      { key: 'buffer_size', label: 'Buffer Size' },
      { key: 'batch_size', label: 'Batch Size' },
      { key: 'tau', label: 'Tau (soft update)' },
      { key: 'gamma', label: 'Discount Factor' },
      { key: 'policy_delay', label: 'Policy Delay' },
      { key: 'target_noise', label: 'Target Policy Noise' },
    ],
  },
  A2C: {
    description: 'Advantage Actor-Critic - Synchronous on-policy algorithm',
    parameters: [
      { key: 'learning_rate', label: 'Learning Rate' },
      { key: 'n_steps', label: 'Steps per Update' },
      { key: 'gamma', label: 'Discount Factor' },
      { key: 'gae_lambda', label: 'GAE Lambda' },
      { key: 'ent_coef', label: 'Entropy Coefficient' },
      { key: 'vf_coef', label: 'Value Function Coef' },
    ],
  },
  DDPG: {
    description: 'Deep Deterministic Policy Gradient - Off-policy algorithm for continuous actions',
    parameters: [
      { key: 'learning_rate', label: 'Learning Rate' },
      { key: 'buffer_size', label: 'Buffer Size' },
      { key: 'batch_size', label: 'Batch Size' },
      { key: 'tau', label: 'Tau (soft update)' },
      { key: 'gamma', label: 'Discount Factor' },
    ],
  },
};

export default function Agents() {
  const [agentList, setAgentList] = useState<APIAgent[]>([]);
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState<APIAgent | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { setActions } = useTopBar();

  useEffect(() => {
    loadAgents();
  }, []);

  useEffect(() => {
    setActions(
      <button
        onClick={openCreate}
        className="flex items-center gap-2 rounded-md bg-notion-blue px-4 py-2 text-sm font-medium text-white hover:opacity-90 transition-opacity"
      >
        <Plus size={16} /> Add Agent
      </button>
    );
    return () => setActions(null);
  }, []);

  const loadAgents = async () => {
    try {
      setLoading(true);
      setError(null);
      const agents = await api.getAgents();
      setAgentList(agents);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load agents');
      console.error('Failed to load agents:', err);
    } finally {
      setLoading(false);
    }
  };

  const openCreate = () => {
    setEditing(null);
    setOpen(true);
  };

  const openEdit = (agent: APIAgent) => {
    setEditing(agent);
    setOpen(true);
  };

  const handleDelete = async (id: number) => {
    if (!confirm('Delete this agent? This action cannot be undone.')) return;

    try {
      await api.deleteAgent(id);
      await loadAgents();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete agent';
      alert(`Error: ${message}`);
    }
  };

  const handleSave = async (data: {
    name: string;
    type: string;
    agent_diameter: number;
    agent_max_speed: number;
    lidar_range: number;
    lidar_rays: number;
    parameters: Record<string, any>
  }) => {
    try {
      if (editing) {
        await api.updateAgent(editing.id, data);
      } else {
        await api.createAgent({ ...data, kinematic_type: 'holonomic' });
      }
      await loadAgents();
      setOpen(false);
      setEditing(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save agent';
      alert(`Error: ${message}`);
    }
  };

  return (
    <div className="space-y-6">
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-notion-red">
          <strong>Error:</strong> {error}
        </div>
      )}

      {loading ? (
        <div className="text-center text-notion-text-secondary py-12">Loading agents...</div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {agentList.length === 0 ? (
            <div className="col-span-2 text-center text-notion-text-secondary py-12">
              No agents found. Click &ldquo;Add Agent&rdquo; to create one.
            </div>
          ) : (
            agentList.map((agent) => {
              const template = agentTemplates[agent.type];
              if (!template) return null;

              return (
                <Card key={agent.id} className="space-y-4">
                  <CardHeader
                    title={agent.name}
                    actions={
                      agent.is_locked ? (
                        <Badge label="Locked" tone="orange" />
                      ) : (
                        <Badge label="Available" tone="green" />
                      )
                    }
                  />
                  <div className="flex flex-wrap items-center gap-3 text-xs text-notion-text-secondary">
                    <span className="rounded-md bg-blue-50 px-3 py-1 text-notion-blue font-medium">{agent.type}</span>
                    <span className="text-notion-text-tertiary">Continuous Action Space</span>
                  </div>

                  {/* Agent Physical Properties */}
                  <div className="rounded-lg border border-notion-border bg-notion-light-gray p-4">
                    <p className="text-sm font-semibold text-notion-text mb-3">Agent Configuration</p>
                    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                      <div className="rounded-md border border-notion-border bg-white px-3 py-2">
                        <p className="text-xs uppercase tracking-wide text-notion-text-secondary">Diameter (m)</p>
                        <p className="font-mono text-sm text-notion-text">{formatValue(agent.agent_diameter)}</p>
                      </div>
                      <div className="rounded-md border border-notion-border bg-white px-3 py-2">
                        <p className="text-xs uppercase tracking-wide text-notion-text-secondary">Max Speed (m/s)</p>
                        <p className="font-mono text-sm text-notion-text">{formatValue(agent.agent_max_speed)}</p>
                      </div>
                      <div className="rounded-md border border-notion-border bg-white px-3 py-2">
                        <p className="text-xs uppercase tracking-wide text-notion-text-secondary">Lidar Range (m)</p>
                        <p className="font-mono text-sm text-notion-text">{formatValue(agent.lidar_range)}</p>
                      </div>
                      <div className="rounded-md border border-notion-border bg-white px-3 py-2">
                        <p className="text-xs uppercase tracking-wide text-notion-text-secondary">Lidar Rays</p>
                        <p className="font-mono text-sm text-notion-text">{agent.lidar_rays}</p>
                      </div>
                    </div>
                  </div>

                  {/* Algorithm Parameters */}
                  <div className="rounded-lg border border-notion-border bg-notion-light-gray p-4">
                    <p className="text-sm font-semibold text-notion-text">Algorithm Parameters</p>
                    <p className="text-xs text-notion-text-secondary">{template.description}</p>
                    <div className="mt-4 grid gap-3 sm:grid-cols-2">
                      {template.parameters.map((param) => (
                        <div key={param.key} className="rounded-md border border-notion-border bg-white px-3 py-2">
                          <p className="text-xs uppercase tracking-wide text-notion-text-secondary">{param.label}</p>
                          <p className="font-mono text-sm text-notion-text">{formatValue(agent.parameters[param.key])}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={() => openEdit(agent)}
                      className="flex flex-1 items-center justify-center gap-2 rounded-md border border-notion-border px-3 py-2 text-sm text-notion-text hover:bg-notion-hover transition-colors"
                      disabled={agent.is_locked}
                    >
                      <PenSquare size={16} /> Edit
                    </button>
                    <button
                      onClick={() => handleDelete(agent.id)}
                      className="flex flex-1 items-center justify-center gap-2 rounded-md border border-notion-border px-3 py-2 text-sm text-notion-red hover:bg-red-50 transition-colors"
                      disabled={agent.is_locked}
                    >
                      <Trash2 size={16} /> Delete
                    </button>
                  </div>
                </Card>
              );
            })
          )}
        </div>
      )}

      <AgentModal
        open={open}
        onClose={() => {
          setOpen(false);
          setEditing(null);
        }}
        onSave={handleSave}
        agent={editing ?? undefined}
      />
    </div>
  );
}
