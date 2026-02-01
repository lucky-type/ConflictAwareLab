import { FormEvent, useEffect, useMemo, useState } from 'react';
import { Plus, X } from 'lucide-react';
import { Card } from '../common/Card';

export type ObstacleBehavior = 'linear' | 'circular' | 'zigzag';

export interface ObstacleConfig {
  id: string;
  diameter: number;
  maxSpeed: number;
  behavior: ObstacleBehavior;
  chaos: number;
}

export interface EnvironmentConfig {
  id: string;
  name: string;
  dimensions: {
    length: number;
    width: number;
    height: number;
  };
  entryBoundary: number;
  exitBoundary: number;
  targetDiameter: number;
  obstacles: ObstacleConfig[];
  isPredicted?: boolean;
}

interface Props {
  open: boolean;
  onClose: () => void;
  onSave: (env: EnvironmentConfig) => void;
  environment?: EnvironmentConfig;
}

const movementLabels: Record<ObstacleBehavior, string> = {
  linear: 'Linear',
  circular: 'Circular',
  zigzag: 'Zigzag',
};

const createId = () =>
  typeof globalThis.crypto !== 'undefined' && 'randomUUID' in globalThis.crypto
    ? globalThis.crypto.randomUUID()
    : Math.random().toString(36).slice(2);

export default function NewEnvironmentModal({ open, onClose, onSave, environment }: Props) {
  const [name, setName] = useState('');
  const [length, setLength] = useState<number | ''>(80);
  const [width, setWidth] = useState<number | ''>(14);
  const [height, setHeight] = useState<number | ''>(6);
  const [entryBoundary, setEntryBoundary] = useState(10);
  const [exitBoundary, setExitBoundary] = useState(90);
  const [targetDiameter, setTargetDiameter] = useState<number | ''>(2);
  const [obstacles, setObstacles] = useState<ObstacleConfig[]>([]);
  const [isPredicted, setIsPredicted] = useState(false);

  useEffect(() => {
    if (!open) return;
    if (environment) {
      setName(environment.name);
      setLength(environment.dimensions.length);
      setWidth(environment.dimensions.width);
      setHeight(environment.dimensions.height);
      setEntryBoundary(environment.entryBoundary);
      setExitBoundary(environment.exitBoundary);
      setTargetDiameter(environment.targetDiameter);
      setObstacles(environment.obstacles.map((obs) => ({ ...obs })));
      setIsPredicted(environment.isPredicted ?? false);
    } else {
      setName('Corridor_Standard');
      setLength(80);
      setWidth(14);
      setHeight(6);
      setEntryBoundary(10);
      setExitBoundary(90);
      setTargetDiameter(2);
      setObstacles([]);
      setIsPredicted(false);
    }
  }, [environment, open]);

  const spawnPercent = entryBoundary;
  const dangerPercent = Math.max(5, exitBoundary - entryBoundary);
  const targetPercent = Math.max(0, 100 - exitBoundary);
  const minGap = 10;

  const handleEntryChange = (value: number) => {
    const next = Math.min(value, exitBoundary - minGap);
    setEntryBoundary(Math.max(0, next));
  };

  const handleExitChange = (value: number) => {
    const next = Math.max(value, entryBoundary + minGap);
    setExitBoundary(Math.min(100, next));
  };

  const previewObstacles = useMemo(() => {
    if (!obstacles.length) return [];
    return obstacles.map((obs, idx) => {
      const ratio = (idx + 1) / (obstacles.length + 1);
      const left = spawnPercent + ratio * dangerPercent;
      const top = 20 + ((idx % 3) / 2) * 40;
      const size = 8 + obs.diameter * 4;
      return { left, top, size, chaos: obs.chaos };
    });
  }, [dangerPercent, obstacles, spawnPercent]);

  const targetPosition = spawnPercent + dangerPercent + targetPercent / 2;

  if (!open) return null;

  const addObstacle = () => {
    setObstacles((prev) => [
      ...prev,
      {
        id: createId(),
        diameter: 1,
        maxSpeed: 4,
        behavior: 'linear',
        chaos: 0.2,
      },
    ]);
  };

  const updateObstacle = (id: string, patch: Partial<ObstacleConfig>) => {
    setObstacles((prev) => prev.map((obs) => (obs.id === id ? { ...obs, ...patch } : obs)));
  };

  const removeObstacle = (id: string) => {
    setObstacles((prev) => prev.filter((obs) => obs.id !== id));
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const payload: EnvironmentConfig = {
      id: environment?.id ?? createId(),
      name,
      dimensions: {
        length: typeof length === 'number' ? length : parseFloat(String(length)) || 80,
        width: typeof width === 'number' ? width : parseFloat(String(width)) || 14,
        height: typeof height === 'number' ? height : parseFloat(String(height)) || 6
      },
      entryBoundary,
      exitBoundary,
      targetDiameter: typeof targetDiameter === 'number' ? targetDiameter : parseFloat(String(targetDiameter)) || 2,
      obstacles,
      isPredicted,
    };
    onSave(payload);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm"
    >
      <div className="w-full max-w-6xl overflow-y-auto max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
        <Card className="p-6">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs uppercase tracking-wide text-notion-blue">
                {environment ? 'Edit Environment' : 'New Environment'}
              </p>
              <h3 className="text-lg font-semibold text-notion-text">{name || 'Unnamed Environment'}</h3>
            </div>
            <button onClick={onClose} className="text-notion-text-secondary hover:text-notion-text">
              <X size={20} />
            </button>
          </div>

          <form onSubmit={submit} className="mt-6 grid gap-6 lg:grid-cols-[1.2fr,1fr]">
            <div className="space-y-5">
              <label className="block space-y-2 text-sm">
                <span className="text-notion-text">Environment Name</span>
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                  placeholder="e.g. Corridor_Test_Range"
                />
              </label>

              <div className="grid grid-cols-3 gap-3 text-sm">
                <label className="space-y-2">
                  <span className="text-notion-text">Length (m)</span>
                  <input
                    type="number"
                    value={length}
                    min={10}
                    max={500}
                    step="any"
                    onChange={(e) => setLength(e.target.value === '' ? '' : parseFloat(e.target.value))}
                    className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                  />
                </label>
                <label className="space-y-2">
                  <span className="text-notion-text">Width (m)</span>
                  <input
                    type="number"
                    value={width}
                    min={2}
                    max={100}
                    step="any"
                    onChange={(e) => setWidth(e.target.value === '' ? '' : parseFloat(e.target.value))}
                    className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                  />
                </label>
                <label className="space-y-2">
                  <span className="text-notion-text">Height (m)</span>
                  <input
                    type="number"
                    value={height}
                    min={2}
                    max={50}
                    step="any"
                    onChange={(e) => setHeight(e.target.value === '' ? '' : parseFloat(e.target.value))}
                    className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                  />
                </label>
              </div>

              <div className="space-y-3 rounded-xl border border-notion-border bg-notion-light-gray p-4 text-sm">
                <div className="flex items-center justify-between text-xs text-notion-text-secondary">
                  <span>Zoning (buffer - danger - finish)</span>
                  <span className="text-notion-text">
                    {Math.round(spawnPercent)}% / {Math.round(dangerPercent)}% / {Math.round(targetPercent)}%
                  </span>
                </div>
                <div className="multi-range relative mt-2">
                  <div className="absolute left-0 right-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-notion-hover" />
                  <div
                    className="absolute left-0 top-1/2 h-1 -translate-y-1/2 rounded-full"
                    style={{
                      width: `${spawnPercent}%`,
                      background: 'rgba(34,211,238,0.6)',
                    }}
                  />
                  <div
                    className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full"
                    style={{
                      width: `${dangerPercent}%`,
                      left: `${spawnPercent}%`,
                      background: 'rgba(251,146,60,0.6)',
                    }}
                  />
                  <div
                    className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full"
                    style={{
                      left: `${spawnPercent + dangerPercent}%`,
                      width: `${targetPercent}%`,
                      background: 'rgba(52,211,153,0.6)',
                    }}
                  />
                  <input
                    type="range"
                    min={5}
                    max={exitBoundary - minGap}
                    value={entryBoundary}
                    onChange={(e) => handleEntryChange(parseFloat(e.target.value))}
                  />
                  <input
                    type="range"
                    min={entryBoundary + minGap}
                    max={95}
                    value={exitBoundary}
                    onChange={(e) => handleExitChange(parseFloat(e.target.value))}
                  />
                </div>
                <p className="text-xs text-notion-text-tertiary">
                  First slider sets the spawn buffer zone, second sets the start of the finish buffer.
                </p>
              </div>

              <label className="block space-y-2 text-sm">
                <span className="text-notion-text">Finish Zone Diameter (m)</span>
                <input
                  type="number"
                  value={targetDiameter}
                  min={0.5}
                  max={50}
                  step={0.1}
                  onChange={(e) => setTargetDiameter(e.target.value === '' ? '' : parseFloat(e.target.value))}
                  className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                />
              </label>

              <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4 text-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-notion-text font-medium">Predicted Mode</span>
                    <p className="text-xs text-notion-text-secondary mt-1">
                      Enable deterministic seeding for reproducible scenarios across experiments
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setIsPredicted(!isPredicted)}
                    className={`relative h-6 w-11 rounded-full transition-colors ${isPredicted ? 'bg-emerald-500' : 'bg-slate-700'
                      }`}
                  >
                    <span
                      className={`absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${isPredicted ? 'translate-x-5' : 'translate-x-0'
                        }`}
                    />
                  </button>
                </div>
                {isPredicted && (
                  <p className="text-xs text-emerald-600 mt-2">
                    ✓ Seed based on creation date + episode number ensures identical scenarios
                  </p>
                )}
              </div>
            </div>

            <div className="space-y-5">
              <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4">
                <div className="flex items-center justify-between text-sm text-notion-text">
                  <span>Scene (side view)</span>
                  <span className="text-notion-text-secondary">{length}m</span>
                </div>
                <div className="relative mt-4 h-56 overflow-hidden rounded-lg border border-notion-border bg-black/30">
                  <div className="absolute inset-6 flex h-[calc(100%-3rem)] items-stretch">
                    <div
                      className="relative h-full rounded-l-lg bg-gradient-to-r from-cyan-500/20 to-cyan-500/10"
                      style={{ width: `${spawnPercent}%` }}
                    >
                      <span className="absolute left-2 top-2 text-xs text-cyan-600">Spawn</span>
                    </div>
                    <div className="relative h-full bg-gradient-to-r from-amber-500/10 via-rose-500/10 to-amber-500/10" style={{ width: `${dangerPercent}%` }}>
                      <span className="absolute left-1/2 top-2 -translate-x-1/2 text-xs text-amber-600">Danger</span>
                      <div className="absolute inset-0">
                        {previewObstacles.map((point, idx) => (
                          <span
                            key={`${point.left}-${idx}`}
                            className="absolute rounded-full bg-rose-400/80 shadow-lg shadow-rose-500/40"
                            style={{
                              width: point.size,
                              height: point.size,
                              left: `${(point.left - spawnPercent) / (dangerPercent || 1) * 100}%`,
                              top: `${point.top}%`,
                              transform: 'translate(-50%, -50%)',
                            }}
                            title={`Chaos ${point.chaos.toFixed(2)}`}
                          />
                        ))}
                      </div>
                    </div>
                    <div
                      className="relative h-full rounded-r-lg bg-gradient-to-r from-emerald-500/20 to-emerald-500/10"
                      style={{ width: `${targetPercent}%` }}
                    >
                      <span className="absolute right-2 top-2 text-xs text-emerald-600">Target</span>
                    </div>
                  </div>
                  <span
                    className="absolute bottom-6 flex h-8 w-8 -translate-x-1/2 items-center justify-center rounded-full border border-emerald-400/60 text-[10px] font-semibold text-emerald-600"
                    style={{ left: `${targetPosition}%` }}
                    title={`Ø ${targetDiameter}м`}
                  >
                    GOAL
                  </span>
                </div>
                <p className="mt-3 text-xs text-notion-text-tertiary">
                  Сегменти та маркери оновлюються миттєво при зміні параметрів.
                </p>
              </div>

              <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4 text-sm">
                <p className="text-notion-text">Statistics</p>
                <ul className="mt-3 space-y-2 text-xs text-notion-text-secondary">
                  <li>Volume: {(Number(length) || 0) * (Number(width) || 0) * (Number(height) || 0)} m³</li>
                  <li>Obstacles: {obstacles.length || 'none'}</li>
                </ul>
              </div>
            </div>

            <div className="lg:col-span-2 space-y-4 rounded-xl border border-notion-border bg-notion-light-gray p-4">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-sm font-semibold text-notion-text">Obstacle Drones</p>
                  <p className="text-xs text-notion-text-secondary">
                    Add drones with different movement strategies and chaos levels.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={addObstacle}
                  className="flex items-center gap-2 rounded-xl border border-notion-blue/50 px-3 py-2 text-xs font-semibold text-notion-blue hover:border-notion-blue"
                >
                  <Plus size={14} /> Add Drone
                </button>
              </div>

              {obstacles.length === 0 && (
                <p className="rounded-xl border border-dashed border-notion-border px-4 py-6 text-center text-sm text-notion-text-tertiary">
                  No obstacles. Click &ldquo;Add Drone&rdquo; to create the first one.
                </p>
              )}

              <div className="space-y-3">
                {obstacles.map((obs, idx) => (
                  <div
                    key={obs.id}
                    className="space-y-3 rounded-xl border border-notion-border bg-white p-4"
                  >
                    <div className="flex items-center justify-between text-xs text-notion-text-secondary">
                      <span>Drone #{idx + 1}</span>
                      <button
                        type="button"
                        onClick={() => removeObstacle(obs.id)}
                        className="text-notion-red hover:text-red-600"
                      >
                        Delete
                      </button>
                    </div>
                    <div className="grid gap-3 text-sm md:grid-cols-2 lg:grid-cols-4">
                      <label className="space-y-1.5">
                        <span className="text-notion-text">Diameter (m)</span>
                        <input
                          type="number"
                          value={obs.diameter}
                          min={0.5}
                          max={20}
                          step={0.1}
                          onChange={(e) =>
                            updateObstacle(obs.id, { diameter: e.target.value === '' ? '' as any : parseFloat(e.target.value) })
                          }
                          className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                        />
                      </label>
                      <label className="space-y-1.5">
                        <span className="text-notion-text">Max Speed (m/s)</span>
                        <input
                          type="number"
                          value={obs.maxSpeed}
                          min={1}
                          max={50}
                          step={0.5}
                          onChange={(e) =>
                            updateObstacle(obs.id, { maxSpeed: e.target.value === '' ? '' as any : parseFloat(e.target.value) })
                          }
                          className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                        />
                      </label>
                      <label className="space-y-1.5">
                        <span className="text-notion-text">Strategy</span>
                        <select
                          value={obs.behavior}
                          onChange={(e) =>
                            updateObstacle(obs.id, { behavior: e.target.value as ObstacleBehavior })
                          }
                          className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                        >
                          {Object.entries(movementLabels).map(([value, label]) => (
                            <option key={value} value={value}>
                              {label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="space-y-1.5">
                        <span className="text-notion-text">Chaos Level ({obs.chaos.toFixed(2)})</span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.05}
                          value={obs.chaos}
                          onChange={(e) => updateObstacle(obs.id, { chaos: parseFloat(e.target.value) })}
                          className="w-full accent-rose-400"
                        />
                      </label>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="lg:col-span-2 flex items-center justify-end gap-3">
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
