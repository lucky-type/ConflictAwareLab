import { FormEvent, useMemo, useState } from 'react';
import { X } from 'lucide-react';
import { Card, CardHeader } from '../common/Card';

interface Props {
  open: boolean;
  onClose: () => void;
}

const hazardPoints = [
  { top: '15%', left: '18%' },
  { top: '32%', left: '46%' },
  { top: '55%', left: '28%' },
  { top: '70%', left: '62%' },
  { top: '24%', left: '72%' },
  { top: '80%', left: '20%' },
  { top: '58%', left: '76%' },
  { top: '40%', left: '12%' },
  { top: '12%', left: '60%' },
  { top: '86%', left: '48%' },
];

export default function NewEnvironmentModal({ open, onClose }: Props) {
  const [name, setName] = useState('Corridor_Standard_v1');
  const [length, setLength] = useState(100);
  const [width, setWidth] = useState(20);
  const [height, setHeight] = useState(10);
  const [zoning, setZoning] = useState(92);
  const [diameter, setDiameter] = useState(1);
  const [lidar, setLidar] = useState(10);

  const hazardLayout = useMemo(
    () =>
      hazardPoints.map((pos, idx) => (
        <span
          key={`${pos.left}-${pos.top}-${idx}`}
          className="absolute h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-rose-500 shadow-lg shadow-rose-500/40"
          style={{ top: pos.top, left: pos.left }}
        />
      )),
    []
  );

  if (!open) return null;

  const submit = (event: FormEvent) => {
    event.preventDefault();
    onClose();
  };

  return (
    <div className="fixed inset-0 z-20 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur">
      <div className="w-full max-w-5xl">
        <Card className="p-6">
          <div className="flex items-start justify-between">
            <CardHeader title="Configure Environment" />
            <button onClick={onClose} className="text-slate-400 hover:text-white">
              <X size={20} />
            </button>
          </div>

          <form onSubmit={submit} className="grid gap-6 lg:grid-cols-2">
            <div className="space-y-4">
              <label className="block space-y-2 text-sm">
                <span className="text-slate-300">Name</span>
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  placeholder="e.g. Narrow_Corridor_v2"
                />
              </label>

              <div className="grid grid-cols-3 gap-3 text-sm">
                <label className="space-y-2">
                  <span className="text-slate-300">Length (m)</span>
                  <input
                    type="number"
                    value={length}
                    onChange={(e) => setLength(parseInt(e.target.value) || 0)}
                    className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  />
                </label>
                <label className="space-y-2">
                  <span className="text-slate-300">Width (m)</span>
                  <input
                    type="number"
                    value={width}
                    onChange={(e) => setWidth(parseInt(e.target.value) || 0)}
                    className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  />
                </label>
                <label className="space-y-2">
                  <span className="text-slate-300">Height (m)</span>
                  <input
                    type="number"
                    value={height}
                    onChange={(e) => setHeight(parseInt(e.target.value) || 0)}
                    className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  />
                </label>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-slate-300">Zoning (start - danger - end)</span>
                  <span className="text-xs text-slate-400">{zoning}%</span>
                </div>
                <div className="flex items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2">
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={zoning}
                    onChange={(e) => setZoning(parseInt(e.target.value))}
                    className="flex-1 accent-cyan-500"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <label className="space-y-2">
                  <span className="text-slate-300">Agent diameter (m)</span>
                  <input
                    type="number"
                    value={diameter}
                    onChange={(e) => setDiameter(parseFloat(e.target.value) || 0)}
                    className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  />
                </label>
                <label className="space-y-2">
                  <span className="text-slate-300">Lidar range (m)</span>
                  <input
                    type="number"
                    value={lidar}
                    onChange={(e) => setLidar(parseFloat(e.target.value) || 0)}
                    className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  />
                </label>
              </div>
            </div>

            <div className="space-y-4">
              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <div className="flex items-center justify-between text-sm text-slate-200">
                  <span>Scene top-down preview</span>
                  <span className="text-slate-400">{length}m</span>
                </div>
                <div className="mt-3 h-56 rounded-lg bg-gradient-to-b from-slate-900 via-slate-900/70 to-slate-900 relative overflow-hidden">
                  <div className="absolute inset-x-6 top-0 bottom-0">
                    <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-cyan-500/10 via-blue-500/10 to-slate-800/60" />
                    {hazardLayout}
                  </div>
                  <div className="absolute left-4 top-4 rounded-full bg-white/10 px-3 py-1 text-xs text-slate-200">spawn</div>
                  <div className="absolute bottom-4 right-4 rounded-full bg-cyan-500/20 px-3 py-1 text-xs text-cyan-100">finish</div>
                </div>
              </div>

              <div className="flex items-center justify-end gap-3">
                <button
                  type="button"
                  onClick={onClose}
                  className="rounded-xl border border-slate-800/80 px-4 py-2 text-sm text-slate-200 hover:border-cyan-400/50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-cyan-500/20"
                >
                  Save environment
                </button>
              </div>
            </div>
          </form>
        </Card>
      </div>
    </div>
  );
}
