import { Activity, Play, Radar, Rocket, Shield } from 'lucide-react';
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  BarChart,
  Bar,
  CartesianGrid,
  AreaChart,
  Area,
} from 'recharts';
import { Badge } from '../components/common/Badge';
import { Card, CardHeader } from '../components/common/Card';
import { metrics, statusCards } from '../data/mockData';

const toneStyles: Record<string, string> = {
  cyan: 'text-cyan-300 bg-cyan-500/10',
  emerald: 'text-emerald-300 bg-emerald-500/10',
  amber: 'text-amber-300 bg-amber-500/10',
  violet: 'text-violet-300 bg-violet-500/10',
};

export default function Dashboard() {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {statusCards.map((card) => (
          <Card key={card.title} className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-wide text-slate-400">{card.title}</p>
                <p className="text-2xl font-semibold text-white">{card.value}</p>
                <p className="text-xs text-slate-400">{card.change}</p>
              </div>
              <div className={`rounded-xl p-3 ${toneStyles[card.tone]}`}>
                <card.icon />
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <Shield size={14} />
              Guardrails active
            </div>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Training reward"
            actions={<Badge label="Live" tone="text-emerald-400 bg-emerald-400/10 border border-emerald-400/30" />}
          />
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metrics.reward}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="step" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1f2937' }} />
                <Line type="monotone" dataKey="value" stroke="#22d3ee" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card>
          <CardHeader title="System health" />
          <div className="flex items-center justify-between rounded-xl bg-slate-900/70 p-4">
            <div>
              <p className="text-xs text-slate-400">Thermal headroom</p>
              <p className="text-3xl font-semibold text-white">86%</p>
              <p className="text-xs text-slate-400">Cooling normalized</p>
            </div>
            <div className="rounded-full bg-emerald-500/10 p-4 text-emerald-400">
              <Radar />
            </div>
          </div>
          <div className="mt-4 h-28">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={metrics.stability}>
                <Area type="monotone" dataKey="value" stroke="#a855f7" fill="url(#stability)" />
                <defs>
                  <linearGradient id="stability" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#a855f7" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#a855f7" stopOpacity={0} />
                  </linearGradient>
                </defs>
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader title="Stability benchmark" />
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metrics.stability}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="step" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1f2937' }} />
                <Bar dataKey="value" fill="#38bdf8" radius={8} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card className="space-y-4">
          <CardHeader title="Simulation queue" />
          <div className="space-y-3">
            {[1, 2, 3].map((item) => (
              <div
                key={item}
                className="flex items-center justify-between rounded-xl border border-slate-800/70 bg-slate-900/60 px-4 py-3"
              >
                <div>
                  <p className="text-sm font-semibold text-white">Mission {item}</p>
                  <p className="text-xs text-slate-400">Dockside stress run</p>
                </div>
                <div className="flex items-center gap-3 text-xs text-slate-400">
                  <Activity size={16} />
                  Running
                </div>
              </div>
            ))}
          </div>
          <button className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-cyan-500/20">
            <Play size={16} /> Start new simulation
          </button>
          <div className="rounded-xl border border-slate-800/70 bg-slate-900/60 px-4 py-3 text-xs text-slate-400">
            <div className="mb-2 flex items-center gap-2 text-sm text-white">
              <Rocket size={16} className="text-cyan-400" />
              Boosted capacity
            </div>
            GPU cluster scaled to 32 A100 nodes for the next 4 hours.
          </div>
        </Card>
      </div>
    </div>
  );
}
