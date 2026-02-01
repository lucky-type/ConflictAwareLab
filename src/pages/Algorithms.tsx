/*
import { useEffect, useState } from 'react';
import { PenSquare, Plus, Trash2 } from 'lucide-react';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import AlgorithmModal from '../components/modals/AlgorithmModal';
import { api, type Algorithm as APIAlgorithm } from '../services/api';

const formatValue = (value: number | string) => {
  if (typeof value === 'number') {
    if (Math.abs(value) >= 1000 || Math.abs(value) < 0.01) {
      return value.toExponential(2);
    }
    return Number.isInteger(value) ? value.toString() : value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
  }
  return value;
};

// Algorithm type definitions with parameters
const algorithmTemplates: Record<string, { description: string; parameters: Array<{ key: string; label: string }> }> = {
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

export default function Algorithms() {
  const [algorithmList, setAlgorithmList] = useState<APIAlgorithm[]>([]);
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState<APIAlgorithm | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadAlgorithms();
  }, []);

  const loadAlgorithms = async () => {
    try {
      setLoading(true);
      setError(null);
      const algos = await api.getAlgorithms();
      setAlgorithmList(algos);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load algorithms');
      console.error('Failed to load algorithms:', err);
    } finally {
      setLoading(false);
    }
  };

  const openCreate = () => {
    setEditing(null);
    setOpen(true);
  };

  const openEdit = (algo: APIAlgorithm) => {
    setEditing(algo);
    setOpen(true);
  };

  const handleDelete = async (id: number) => {
    if (!confirm('Delete this algorithm? This action cannot be undone.')) return;

    try {
      await api.deleteAlgorithm(id);
      await loadAlgorithms();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete algorithm';
      alert(`Error: ${message}`);
    }
  };

  const handleSave = async (data: { name: string; type: string; parameters: Record<string, any> }) => {
    try {
      if (editing) {
        await api.updateAlgorithm(editing.id, data);
      } else {
        await api.createAlgorithm(data);
      }
      await loadAlgorithms();
      setOpen(false);
      setEditing(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save algorithm';
      alert(`Error: ${message}`);
    }
  };

  return (
    <div className="space-y-6">
      {error && (
        <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-600">
          <strong>Error:</strong> {error}
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-wide text-notion-blue">Policy stack</p>
          <h2 className="text-xl font-semibold text-notion-text">Algorithms</h2>
        </div>
        <button
          onClick={openCreate}
          className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-notion-blue to-notion-blue px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-notion-blue/20"
        >
          <Plus size={16} /> Add Algorithm
        </button>
      </div>

      {loading ? (
        <div className="text-center text-notion-text-secondary py-12">Loading algorithms...</div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {algorithmList.length === 0 ? (
            <div className="col-span-2 text-center text-notion-text-secondary py-12">
              No algorithms found. Click &ldquo;Add Algorithm&rdquo; to create one.
            </div>
          ) : (
            algorithmList.map((algo) => {
              const template = algorithmTemplates[algo.type];
              if (!template) return null;
              
              return (
                <Card key={algo.id} className="space-y-4">
                <CardHeader
                  title={algo.name}
                  actions={
                    algo.is_locked ? (
                      <Badge label="Locked" tone="orange" />
                    ) : (
                      <Badge label="Available" tone="green" />
                    )
                  }
                />
                <div className="flex flex-wrap items-center gap-3 text-xs text-notion-text-secondary">
                  <span className="rounded-full border border-cyan-500/40 px-3 py-1 text-cyan-600">{algo.type}</span>
                  <span className="text-notion-text-tertiary">Continuous Action Space</span>
                </div>

              <div className="rounded-xl border border-notion-border/60 bg-slate-950/30 p-4">
                <p className="text-sm font-semibold text-notion-text">Parameters</p>
                <p className="text-xs text-notion-text-secondary">{template.description}</p>
                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                  {template.parameters.map((param) => (
                    <div key={param.key} className="rounded-lg border border-notion-border bg-notion-light-gray px-3 py-2">
                      <p className="text-xs uppercase tracking-wide text-notion-text-secondary">{param.label}</p>
                      <p className="font-mono text-sm text-notion-text">{formatValue(algo.parameters[param.key])}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={() => openEdit(algo)}
                  className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-notion-border px-3 py-2 text-sm text-notion-text hover:border-cyan-500/50"
                >
                  <PenSquare size={16} /> Edit
                </button>
                <button
                  onClick={() => handleDelete(algo.id)}
                  className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-rose-500/40 px-3 py-2 text-sm text-rose-600 hover:border-rose-400/80"
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

      <AlgorithmModal
        open={open}
        onClose={() => {
          setOpen(false);
          setEditing(null);
        }}
        onSave={handleSave}
        algorithm={editing ?? undefined}
      />
    </div>
  );
}
*/
export default function Algorithms() { return null; }
