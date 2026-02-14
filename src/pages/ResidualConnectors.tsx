import { useEffect, useState } from 'react';
import { PenSquare, Plus, Trash2, Layers3 } from 'lucide-react';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import ResidualConnectorModal from '../components/modals/ResidualConnectorModal';
import { api, type ResidualConnector } from '../services/api';
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

// Algorithm type definitions with parameters
const algorithmTemplates: Record<string, { description: string; parameters: Array<{ key: string; label: string }> }> = {
  PPO: {
    description: 'Proximal Policy Optimization - On-policy residual learner',
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
  A2C: {
    description: 'Advantage Actor-Critic - Synchronous residual learner',
    parameters: [
      { key: 'learning_rate', label: 'Learning Rate' },
      { key: 'n_steps', label: 'Steps per Update' },
      { key: 'gamma', label: 'Discount Factor' },
      { key: 'gae_lambda', label: 'GAE Lambda' },
      { key: 'ent_coef', label: 'Entropy Coefficient' },
      { key: 'vf_coef', label: 'Value Function Coef' },
    ],
  },
};

export default function ResidualConnectors() {
  const [connectorList, setConnectorList] = useState<ResidualConnector[]>([]);
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState<ResidualConnector | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { setActions } = useTopBar();

  useEffect(() => {
    loadConnectors();
  }, []);

  useEffect(() => {
    setActions(
      <button
        onClick={openCreate}
        className="flex items-center gap-2 rounded-md bg-notion-blue px-4 py-2 text-sm font-medium text-white hover:opacity-90 transition-opacity"
      >
        <Plus size={16} /> Add Residual Connector
      </button>
    );
    return () => setActions(null);
  }, []);

  const loadConnectors = async () => {
    try {
      setLoading(true);
      setError(null);
      const connectors = await api.getResidualConnectors();
      setConnectorList(connectors);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load residual connectors');
      console.error('Failed to load residual connectors:', err);
    } finally {
      setLoading(false);
    }
  };

  const openCreate = () => {
    setEditing(null);
    setOpen(true);
  };

  const openEdit = (connector: ResidualConnector) => {
    setEditing(connector);
    setOpen(true);
  };

  const handleDelete = async (id: number) => {
    if (!confirm('Delete this residual connector? This action cannot be undone.')) return;

    try {
      await api.deleteResidualConnector(id);
      await loadConnectors();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete residual connector';
      alert(`Error: ${message}`);
    }
  };

  const handleSave = async (data: {
    name: string;
    parent_agent_id: number;
    algorithm: string;
    net_arch: number[];
    parameters: Record<string, any>;
    k_factor: number;
    adaptive_k: boolean;
  }) => {
    try {
      if (editing) {
        await api.updateResidualConnector(editing.id, data);
      } else {
        await api.createResidualConnector(data);
      }
      await loadConnectors();
      setOpen(false);
      setEditing(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save residual connector';
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
        <div className="text-center text-notion-text-secondary py-12">Loading residual connectors...</div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {connectorList.length === 0 ? (
            <div className="col-span-2 text-center text-notion-text-secondary py-12">
              No residual connectors found. Click &ldquo;Add Residual Connector&rdquo; to create one.
            </div>
          ) : (
            connectorList.map((connector) => {
              const template = algorithmTemplates[connector.algorithm];
              if (!template) return null;

              return (
                <Card key={connector.id} className="space-y-4">
                  <CardHeader
                    title={
                      <div className="flex items-center gap-2">
                        <Layers3 size={20} className="text-notion-purple" />
                        <span>{connector.name}</span>
                      </div>
                    }
                    actions={
                      connector.is_locked ? (
                        <Badge label="Locked" tone="orange" />
                      ) : (
                        <Badge label="Available" tone="green" />
                      )
                    }
                  />
                  <div className="flex flex-wrap items-center gap-3 text-xs text-notion-text-secondary">
                    <span className="rounded-md bg-purple-50 px-3 py-1 text-notion-purple font-medium">{connector.algorithm}</span>
                    <span className="text-notion-text-tertiary">Safety Corrector Layer</span>
                  </div>

                  {/* Parent Agent Info */}
                  {connector.parent_agent && (
                    <div className="rounded-lg border border-notion-border bg-notion-light-gray p-4">
                      <p className="text-sm font-semibold text-notion-text mb-3">Inherited from Parent Agent</p>
                      <div className="flex items-center gap-2 mb-3">
                        <span className="text-xs text-notion-text-secondary">Base Model:</span>
                        <span className="rounded-md bg-blue-50 px-2 py-0.5 text-xs text-notion-blue font-medium">
                          {connector.parent_agent.name}
                        </span>
                      </div>
                      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                        <div className="rounded-md border border-notion-border bg-white px-3 py-2">
                          <p className="text-xs uppercase tracking-wide text-notion-text-secondary">Diameter (m)</p>
                          <p className="font-mono text-sm text-notion-text">{formatValue(connector.parent_agent.agent_diameter)}</p>
                        </div>
                        <div className="rounded-md border border-notion-border bg-white px-3 py-2">
                          <p className="text-xs uppercase tracking-wide text-notion-text-secondary">Max Speed (m/s)</p>
                          <p className="font-mono text-sm text-notion-text">{formatValue(connector.parent_agent.agent_max_speed)}</p>
                        </div>
                        <div className="rounded-md border border-notion-border bg-white px-3 py-2">
                          <p className="text-xs uppercase tracking-wide text-notion-text-secondary">Lidar Range (m)</p>
                          <p className="font-mono text-sm text-notion-text">{formatValue(connector.parent_agent.lidar_range)}</p>
                        </div>
                        <div className="rounded-md border border-notion-border bg-white px-3 py-2">
                          <p className="text-xs uppercase tracking-wide text-notion-text-secondary">Lidar Rays</p>
                          <p className="font-mono text-sm text-notion-text">{connector.parent_agent.lidar_rays}</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Residual Network Architecture */}
                  <div className="rounded-lg border border-purple-200 bg-purple-50 p-4">
                    <p className="text-sm font-semibold text-notion-text">Residual Network Architecture</p>
                    <p className="text-xs text-notion-text-secondary mb-3">Neural network for safety correction</p>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <div className="rounded-md border border-purple-200 bg-white px-3 py-2">
                        <p className="text-xs uppercase tracking-wide text-notion-purple">Network Layers</p>
                        <p className="font-mono text-sm text-notion-text">[{connector.net_arch.join(', ')}]</p>
                      </div>
                      <div className="rounded-md border border-purple-200 bg-white px-3 py-2">
                        <p className="text-xs uppercase tracking-wide text-notion-purple">K-Factor</p>
                        <p className="font-mono text-sm text-notion-text">{(connector as any).k_factor ?? 0.15}</p>
                      </div>
                    </div>
                    {/* {(connector as any).adaptive_k && (
                      <div className="mt-3 flex items-center gap-2 rounded-md border border-blue-200 bg-blue-50 px-3 py-2">
                        <div className="w-2 h-2 rounded-full bg-notion-blue animate-pulse"></div>
                        <p className="text-xs text-notion-blue">CARS enabled - K smoothly scales based on conflict and risk (λ)</p>
                      </div>
                    )} */}
                  </div>

                  {/* Algorithm Parameters */}
                  <div className="rounded-lg border border-notion-border bg-notion-light-gray p-4">
                    <p className="text-sm font-semibold text-notion-text">Residual Algorithm Parameters</p>
                    <p className="text-xs text-notion-text-secondary">{template.description}</p>
                    <div className="mt-4 grid gap-3 sm:grid-cols-2">
                      {template.parameters.map((param) => (
                        <div key={param.key} className="rounded-md border border-notion-border bg-white px-3 py-2">
                          <p className="text-xs uppercase tracking-wide text-notion-text-secondary">{param.label}</p>
                          <p className="font-mono text-sm text-notion-text">{formatValue(connector.parameters[param.key])}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={() => openEdit(connector)}
                      className="flex flex-1 items-center justify-center gap-2 rounded-md border border-notion-border px-3 py-2 text-sm text-notion-text hover:bg-notion-hover transition-colors"
                      disabled={connector.is_locked}
                    >
                      <PenSquare size={16} /> Edit
                    </button>
                    <button
                      onClick={() => handleDelete(connector.id)}
                      className="flex flex-1 items-center justify-center gap-2 rounded-md border border-notion-border px-3 py-2 text-sm text-notion-red hover:bg-red-50 transition-colors"
                      disabled={connector.is_locked}
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

      <ResidualConnectorModal
        open={open}
        onClose={() => {
          setOpen(false);
          setEditing(null);
        }}
        onSave={handleSave}
        connector={editing ?? undefined}
      />
    </div>
  );
}
