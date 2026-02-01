import { useEffect, useMemo, useState } from 'react';
import { Pencil, Plus, Search, Trash2 } from 'lucide-react';
import { Card, CardHeader } from '../components/common/Card';
import { Table, TableBody, TableCell, TableHead, TableRow } from '../components/common/Table';
import NewEnvironmentModal, { EnvironmentConfig } from '../components/modals/NewEnvironmentModal';
import { api, type Environment as APIEnvironment } from '../services/api';
import { useTopBar } from '../context/TopBarContext';

// Helper functions to convert between API and UI formats
const apiToUI = (env: APIEnvironment): EnvironmentConfig => ({
  id: env.id.toString(),
  name: env.name,
  dimensions: {
    length: env.length,
    width: env.width,
    height: env.height,
  },
  entryBoundary: env.buffer_start * 100, // Convert 0-1 to percentage
  exitBoundary: env.buffer_end * 100,
  targetDiameter: env.target_diameter,
  obstacles: env.obstacles.map((obs: any) => ({
    id: Math.random().toString(36).slice(2),
    diameter: obs.diameter,
    maxSpeed: obs.speed,
    behavior: obs.strategy,
    chaos: obs.chaos,
  })),
  isPredicted: env.is_predicted,
});

const uiToAPI = (env: EnvironmentConfig) => ({
  name: env.name,
  length: env.dimensions.length,
  width: env.dimensions.width,
  height: env.dimensions.height,
  buffer_start: env.entryBoundary / 100, // Convert percentage to 0-1
  buffer_end: env.exitBoundary / 100,
  target_diameter: env.targetDiameter,
  obstacles: env.obstacles.map((obs) => ({
    type: 'dynamic',
    diameter: obs.diameter,
    speed: obs.maxSpeed,
    strategy: obs.behavior,
    chaos: obs.chaos,
  })),
  is_predicted: env.isPredicted ?? false,
});

export default function Environments() {
  const [search, setSearch] = useState('');
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState<EnvironmentConfig | null>(null);
  const [environmentList, setEnvironmentList] = useState<EnvironmentConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { setActions } = useTopBar();

  // Set TopBar actions
  useEffect(() => {
    setActions(
      <button
        onClick={openCreator}
        className="flex items-center gap-2 rounded-md bg-notion-blue px-4 py-2 text-sm font-medium text-white hover:opacity-90 transition-opacity"
      >
        <Plus size={16} /> Add Environment
      </button>
    );
    return () => setActions(null);
  }, []);

  // Load environments from API
  useEffect(() => {
    loadEnvironments();
  }, []);

  const loadEnvironments = async () => {
    try {
      setLoading(true);
      setError(null);
      const envs = await api.getEnvironments();
      setEnvironmentList(envs.map(apiToUI));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load environments');
      console.error('Failed to load environments:', err);
    } finally {
      setLoading(false);
    }
  };

  const filtered = useMemo(() => {
    if (!search.trim()) return environmentList;
    return environmentList.filter((env) => env.name.toLowerCase().includes(search.toLowerCase()));
  }, [environmentList, search]);

  const zoneSummary = (env: EnvironmentConfig) => {
    const spawn = Math.round(env.entryBoundary);
    const danger = Math.round(env.exitBoundary - env.entryBoundary);
    const target = Math.round(100 - env.exitBoundary);
    return `${spawn}% / ${danger}% / ${target}%`;
  };

  const handleSave = async (payload: EnvironmentConfig) => {
    try {
      setError(null);
      const apiData = uiToAPI(payload);

      if (editing) {
        // Update existing
        await api.updateEnvironment(parseInt(payload.id), apiData);
      } else {
        // Create new
        await api.createEnvironment(apiData);
      }

      // Reload the list
      await loadEnvironments();
      setEditing(null);
      setOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save environment');
      console.error('Failed to save environment:', err);
      alert(err instanceof Error ? err.message : 'Failed to save environment');
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this environment? This action cannot be undone.')) return;

    try {
      setError(null);
      await api.deleteEnvironment(parseInt(id));
      await loadEnvironments();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete environment');
      console.error('Failed to delete environment:', err);
      alert(err instanceof Error ? err.message : 'Failed to delete environment');
    }
  };

  const openCreator = () => {
    setEditing(null);
    setOpen(true);
  };

  const openEditor = (env: EnvironmentConfig) => {
    setEditing(env);
    setOpen(true);
  };

  return (
    <div className="space-y-6">
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-notion-red">
          <strong>Error:</strong> {error}
        </div>
      )}

      <Card>
        <CardHeader
          title="Environment List"
          actions={
            <div className="flex items-center gap-2">
              <label className="flex items-center gap-2 rounded-md border border-notion-border bg-white px-3 py-2 text-sm text-notion-text">
                <Search size={16} className="text-notion-text-tertiary" />
                <input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search by name..."
                  className="bg-transparent text-sm text-notion-text placeholder:text-notion-text-tertiary focus:outline-none"
                />
              </label>
              <button
                onClick={openCreator}
                className="hidden items-center gap-2 rounded-md border border-notion-border px-3 py-2 text-xs font-medium text-notion-text hover:bg-notion-hover sm:flex"
              >
                <Plus size={14} /> Add
              </button>
            </div>
          }
        />

        {loading ? (
          <div className="p-8 text-center text-notion-text-secondary">Loading environments...</div>
        ) : filtered.length === 0 ? (
          <div className="p-8 text-center text-notion-text-secondary">
            {search ? 'No environments match your search' : 'No environments yet. Create one to get started.'}
          </div>
        ) : (
          <Table>
            <TableHead>
              <TableRow>
                <TableCell className="text-xs font-semibold uppercase tracking-wide text-notion-text-secondary">Name</TableCell>
                <TableCell className="text-xs font-semibold uppercase tracking-wide text-notion-text-secondary">Corridor</TableCell>
                <TableCell className="text-xs font-semibold uppercase tracking-wide text-notion-text-secondary">Zones</TableCell>
                <TableCell className="text-xs font-semibold uppercase tracking-wide text-notion-text-secondary">Target</TableCell>
                <TableCell className="text-xs font-semibold uppercase tracking-wide text-notion-text-secondary">Obstacles</TableCell>
                <TableCell className="text-right text-xs font-semibold uppercase tracking-wide text-notion-text-secondary">
                  Actions
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filtered.map((env) => (
                <TableRow key={env.id}>
                  <TableCell>
                    <div className="font-semibold text-notion-text flex items-center gap-2">
                      {env.name}
                      {env.isPredicted && (
                        <span className="px-1.5 py-0.5 text-[10px] bg-green-50 text-notion-green rounded-md">
                          PREDICTED
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-notion-text-secondary">{env.obstacles.length} obstacle(s)</p>
                  </TableCell>
                  <TableCell>
                    <div className="text-sm text-notion-text">
                      {env.dimensions.length}×{env.dimensions.width}×{env.dimensions.height} m
                    </div>
                    <p className="text-xs text-notion-text-tertiary">Length × Width × Height</p>
                  </TableCell>
                  <TableCell>
                    <div className="text-sm text-notion-text">{zoneSummary(env)}</div>
                    <div className="mt-2 flex h-1.5 w-32 overflow-hidden rounded-full border border-notion-border">
                      <span style={{ width: `${env.entryBoundary}%` }} className="block bg-notion-blue" />
                      <span
                        style={{ width: `${env.exitBoundary - env.entryBoundary}%` }}
                        className="block bg-notion-orange"
                      />
                      <span style={{ width: `${100 - env.exitBoundary}%` }} className="block bg-notion-green" />
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="text-sm text-notion-text">Ø {env.targetDiameter} m</div>
                    <p className="text-xs text-notion-text-tertiary">Target zone</p>
                  </TableCell>
                  <TableCell>
                    <div className="text-sm text-notion-text">
                      {env.obstacles.length ? `${env.obstacles.length} drone(s)` : 'No drones'}
                    </div>
                    {env.obstacles.length > 0 && (
                      <p className="text-xs text-notion-text-tertiary">
                        {env.obstacles
                          .slice(0, 2)
                          .map((obs) => obs.behavior)
                          .join(', ')}
                        {env.obstacles.length > 2 && '...'}
                      </p>
                    )}
                  </TableCell>
                  <TableCell className="text-right text-notion-text-secondary">
                    <div className="flex items-center justify-end gap-2">
                      <button
                        className="rounded-md border border-notion-border p-2 hover:bg-notion-hover hover:text-notion-text transition-colors"
                        onClick={() => openEditor(env)}
                      >
                        <Pencil size={16} />
                        <span className="sr-only">Edit</span>
                      </button>
                      <button
                        className="rounded-md border border-notion-border p-2 text-notion-red hover:bg-red-50 transition-colors"
                        onClick={() => handleDelete(env.id)}
                      >
                        <Trash2 size={16} />
                        <span className="sr-only">Delete</span>
                      </button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Card>

      <div className="flex justify-end">
        <button
          onClick={openCreator}
          className="flex items-center gap-2 rounded-md border border-dashed border-notion-border px-5 py-3 text-sm font-medium text-notion-text-secondary hover:border-notion-blue hover:text-notion-blue transition-colors"
        >
          <Plus size={16} /> Add
        </button>
      </div>

      <NewEnvironmentModal
        open={open}
        onClose={() => {
          setOpen(false);
          setEditing(null);
        }}
        onSave={handleSave}
        environment={editing ?? undefined}
      />
    </div>
  );
}
