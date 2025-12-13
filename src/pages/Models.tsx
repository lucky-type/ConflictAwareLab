import { BrainCircuit, CloudDownload, FolderPlus, Shield } from 'lucide-react';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { models, statuses } from '../data/mockData';

export default function Models() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-cyan-300">Registry</p>
          <h2 className="text-xl font-semibold text-white">Models</h2>
        </div>
        <div className="flex gap-2">
          <button className="flex items-center gap-2 rounded-xl border border-slate-800/80 px-4 py-2 text-sm text-slate-200 hover:border-cyan-500/50">
            <FolderPlus size={16} /> New model
          </button>
          <button className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-cyan-500/20">
            <CloudDownload size={16} /> Pull checkpoint
          </button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {models.map((model) => (
          <Card key={model.name} className="space-y-3">
            <CardHeader
              title={model.name}
              actions={<Badge label={model.status} tone={statuses[model.status as keyof typeof statuses]} />}
            />
            <p className="text-sm text-slate-300">Owner: {model.owner}</p>
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>{model.size}</span>
              <span>{model.metric}</span>
            </div>
            <div className="flex flex-wrap gap-2 text-xs text-slate-300">
              {model.tags.map((tag) => (
                <span key={tag} className="rounded-full bg-slate-800 px-3 py-1">
                  {tag}
                </span>
              ))}
            </div>
            <div className="flex items-center gap-2 text-xs text-emerald-400">
              <Shield size={14} /> Guardrails validated
            </div>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader title="Latency audit" />
        <div className="grid gap-3 md:grid-cols-4">
          {[12, 19, 22, 18].map((latency, idx) => (
            <div
              key={idx}
              className="rounded-xl border border-slate-800/70 bg-slate-900/70 px-4 py-3 text-center text-sm text-slate-200"
            >
              <div className="mb-1 flex items-center justify-center gap-2 text-cyan-300">
                <BrainCircuit size={16} />
                Path {idx + 1}
              </div>
              <p className="text-2xl font-semibold text-white">{latency}ms</p>
              <p className="text-xs text-slate-400">P99 inference</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
