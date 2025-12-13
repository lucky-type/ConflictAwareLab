import { useState } from 'react';
import { PenSquare, ShieldCheck } from 'lucide-react';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { rewardFunctions, statuses } from '../data/mockData';
import NewRewardFunctionModal from '../components/modals/NewRewardFunctionModal';

export default function RewardFunctions() {
  const [open, setOpen] = useState(false);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-cyan-300">Control rewards</p>
          <h2 className="text-xl font-semibold text-white">Reward functions</h2>
        </div>
        <button
          onClick={() => setOpen(true)}
          className="flex items-center gap-2 rounded-xl border border-slate-800/80 px-4 py-2 text-sm text-slate-200 hover:border-cyan-500/50"
        >
          <PenSquare size={16} /> Draft new
        </button>
      </div>

      <Card className="space-y-4">
        <CardHeader title="Definitions" />
        <div className="space-y-3">
          {rewardFunctions.map((reward) => (
            <div
              key={reward.name}
              className="flex items-center justify-between rounded-xl border border-slate-800/60 bg-slate-900/70 px-4 py-3"
            >
              <div>
                <p className="text-sm font-semibold text-white">{reward.name}</p>
                <p className="text-xs text-slate-400">Maintainer: {reward.author}</p>
              </div>
              <div className="flex items-center gap-3">
                <Badge label={reward.status} tone={statuses[reward.status as keyof typeof statuses]} />
                <span className="text-xs text-slate-400">Updated {reward.lastUpdated}</span>
                <div className="flex items-center gap-2 rounded-full border border-slate-800/80 bg-slate-900/70 px-3 py-1 text-xs text-emerald-400">
                  <ShieldCheck size={14} /> Safe rollout
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <NewRewardFunctionModal open={open} onClose={() => setOpen(false)} />
    </div>
  );
}
