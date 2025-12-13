import { AlertTriangle, CheckCircle, Plus, Wifi } from 'lucide-react';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { environments, statuses } from '../data/mockData';

export default function Environments() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-cyan-300">Mission spaces</p>
          <h2 className="text-xl font-semibold text-white">Environment catalog</h2>
        </div>
        <button className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-cyan-500/20">
          <Plus size={16} /> New environment
        </button>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {environments.map((env) => (
          <Card key={env.name} className="space-y-3">
            <CardHeader
              title={env.name}
              actions={<Badge label={env.status} tone={statuses[env.status as keyof typeof statuses]} />}
            />
            <p className="text-sm text-slate-300">{env.description}</p>
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>{env.drones}</span>
              <span>Reward {env.reward}</span>
              <span>Uptime {env.uptime}</span>
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-400">
              {env.status === 'Degraded' ? (
                <AlertTriangle size={16} className="text-amber-400" />
              ) : (
                <CheckCircle size={16} className="text-emerald-400" />
              )}
              {env.status === 'Degraded' ? 'Mitigations applied' : 'Nominal'}
            </div>
          </Card>
        ))}
      </div>

      <Card className="space-y-4">
        <CardHeader
          title="Live control"
          actions={<Badge label="Streaming" tone="text-emerald-400 bg-emerald-400/10 border border-emerald-400/30" />}
        />
        <div className="grid gap-4 md:grid-cols-3">
          {[1, 2, 3].map((id) => (
            <div key={id} className="rounded-xl border border-slate-800/70 bg-slate-900/70 p-4">
              <div className="mb-3 flex items-center justify-between text-sm text-white">
                <span>Drone #{id}</span>
                <Wifi size={16} className="text-cyan-400" />
              </div>
              <div className="h-24 rounded-lg bg-gradient-to-br from-cyan-500/10 via-blue-500/5 to-slate-800/60" />
              <p className="mt-3 text-xs text-slate-400">Streaming telemetry with low latency</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
