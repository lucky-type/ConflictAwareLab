import { Gauge, PlayCircle, RefreshCw } from 'lucide-react';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { algorithms, statuses } from '../data/mockData';

export default function Algorithms() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-cyan-300">Policy stack</p>
          <h2 className="text-xl font-semibold text-white">Algorithms</h2>
        </div>
        <button className="flex items-center gap-2 rounded-xl border border-slate-800/80 px-4 py-2 text-sm text-slate-200 hover:border-cyan-500/50">
          <RefreshCw size={16} /> Sync from registry
        </button>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {algorithms.map((algo) => (
          <Card key={algo.name} className="space-y-3">
            <CardHeader
              title={`${algo.name} (${algo.version})`}
              actions={<Badge label={algo.status} tone={statuses[algo.status as keyof typeof statuses]} />}
            />
            <p className="text-sm text-slate-300">Last run {algo.lastRun}</p>
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>{algo.throughput}</span>
              <div className="flex items-center gap-2 text-emerald-400">
                <Gauge size={16} /> Stabilized
              </div>
            </div>
            <div className="flex gap-2">
              <button className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-slate-800/80 px-3 py-2 text-xs text-slate-200 hover:border-cyan-500/50">
                <PlayCircle size={16} /> Deploy
              </button>
              <button className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-slate-800/80 px-3 py-2 text-xs text-slate-200 hover:border-cyan-500/50">
                <RefreshCw size={16} /> Rollback
              </button>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
