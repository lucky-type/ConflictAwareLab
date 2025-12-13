import { useState } from 'react';
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Badge } from '../components/common/Badge';
import { Card, CardHeader } from '../components/common/Card';
import { Table, TableBody, TableCell, TableHead, TableRow } from '../components/common/Table';
import NewExperimentModal from '../components/modals/NewExperimentModal';
import { comparison, experiments, metrics, statuses } from '../data/mockData';
import { Play, Plus, Timer, Waves } from 'lucide-react';

export default function Experiments() {
  const [open, setOpen] = useState(false);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-cyan-300">Trials</p>
          <h2 className="text-xl font-semibold text-white">Experiments</h2>
        </div>
        <button
          onClick={() => setOpen(true)}
          className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-cyan-500/20"
        >
          <Plus size={16} /> New experiment
        </button>
      </div>

      <Card className="space-y-4">
        <CardHeader title="Runs" />
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Title</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Environment</TableCell>
              <TableCell>Algorithm</TableCell>
              <TableCell>Reward</TableCell>
              <TableCell>Owner</TableCell>
              <TableCell>Duration</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {experiments.map((exp) => (
              <TableRow key={exp.id}>
                <TableCell className="font-semibold text-white">{exp.id}</TableCell>
                <TableCell className="text-slate-200">{exp.title}</TableCell>
                <TableCell>
                  <Badge label={exp.status} tone={statuses[exp.status as keyof typeof statuses]} />
                </TableCell>
                <TableCell className="text-slate-200">{exp.env}</TableCell>
                <TableCell className="text-slate-200">{exp.algorithm}</TableCell>
                <TableCell className="text-emerald-300">{exp.reward}</TableCell>
                <TableCell className="text-slate-200">{exp.owner}</TableCell>
                <TableCell className="text-slate-200">{exp.duration}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title="Comparison" />
          <div className="space-y-3">
            {comparison.map((row) => (
              <div key={row.metric} className="flex items-center justify-between rounded-xl border border-slate-800/70 bg-slate-900/60 px-4 py-3">
                <div>
                  <p className="text-sm font-semibold text-white">{row.metric}</p>
                  <p className="text-xs text-slate-400">Baseline vs candidate</p>
                </div>
                <div className="text-right text-sm">
                  <p className="text-slate-400">Baseline {row.baseline}</p>
                  <p className="text-emerald-400">Candidate {row.candidate}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
        <Card>
          <CardHeader title="Running simulation" />
          <div className="space-y-3">
            <div className="rounded-xl border border-slate-800/70 bg-slate-900/60 p-4">
              <div className="flex items-center justify-between text-sm text-white">
                <span>EXP-2412-A</span>
                <Badge label="Running" tone={statuses.Running} />
              </div>
              <p className="mt-2 text-xs text-slate-400">Warehouse navigation rollout</p>
              <div className="mt-4 flex items-center gap-3 text-xs text-slate-400">
                <Timer size={16} /> Elapsed 01:18:42
              </div>
              <div className="mt-4 h-32">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metrics.reward}>
                    <XAxis dataKey="step" hide />
                    <YAxis hide />
                    <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1f2937' }} />
                    <Line type="monotone" dataKey="value" stroke="#22d3ee" strokeWidth={3} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-3 flex items-center gap-2 text-emerald-400">
                <Waves size={16} /> Stable trajectory
              </div>
            </div>
            <div className="flex gap-2">
              <button className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-slate-800/80 px-3 py-2 text-sm text-slate-200 hover:border-cyan-500/50">
                Pause
              </button>
              <button className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 px-3 py-2 text-sm font-semibold text-white shadow-lg shadow-cyan-500/20">
                <Play size={16} /> Resume
              </button>
            </div>
          </div>
        </Card>
      </div>

      <NewExperimentModal open={open} onClose={() => setOpen(false)} />
    </div>
  );
}
