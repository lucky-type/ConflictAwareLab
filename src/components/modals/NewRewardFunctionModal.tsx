import { FormEvent, useState } from 'react';
import { X } from 'lucide-react';
import { Card, CardHeader } from '../common/Card';

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function NewRewardFunctionModal({ open, onClose }: Props) {
  const [wProg, setWProg] = useState(2);
  const [wTime, setWTime] = useState(1);
  const [wJerk, setWJerk] = useState(1);
  const [name, setName] = useState('Standard Racing');

  if (!open) return null;

  const submit = (event: FormEvent) => {
    event.preventDefault();
    onClose();
  };

  return (
    <div className="fixed inset-0 z-20 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur">
      <div className="w-full max-w-3xl">
        <Card className="p-6">
          <div className="flex items-start justify-between">
            <CardHeader title="New Reward Function" />
            <button onClick={onClose} className="text-slate-400 hover:text-white">
              <X size={20} />
            </button>
          </div>

          <form onSubmit={submit} className="grid gap-6 md:grid-cols-2">
            <div className="space-y-4">
              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 text-sm text-slate-200">
                <p className="font-mono text-xs text-slate-300">R = (w_prog × dist) - (w_time × t) - (w_jerk × j)</p>
                <p className="mt-2 text-slate-400">Adjust weights to balance progress, time and jerk penalties.</p>
              </div>

              <label className="block space-y-2 text-sm">
                <span className="text-slate-300">Function name</span>
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  placeholder="e.g. Agile Corridor"
                />
              </label>
            </div>

            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-3 text-sm">
                <label className="space-y-2">
                  <span className="text-slate-300">w_prog</span>
                  <input
                    type="number"
                    value={wProg}
                    onChange={(e) => setWProg(parseFloat(e.target.value) || 0)}
                    className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  />
                </label>
                <label className="space-y-2">
                  <span className="text-slate-300">w_time</span>
                  <input
                    type="number"
                    value={wTime}
                    onChange={(e) => setWTime(parseFloat(e.target.value) || 0)}
                    className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  />
                </label>
                <label className="space-y-2">
                  <span className="text-slate-300">w_jerk</span>
                  <input
                    type="number"
                    value={wJerk}
                    onChange={(e) => setWJerk(parseFloat(e.target.value) || 0)}
                    className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  />
                </label>
              </div>

              <div className="flex items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/70 p-3 text-xs">
                <div className="flex-1 rounded-lg bg-emerald-500/10 px-3 py-2 text-emerald-300">TARGET REACHED <span className="text-white">+1000</span></div>
                <div className="flex-1 rounded-lg bg-rose-500/10 px-3 py-2 text-rose-300">CRASH <span className="text-white">-1000</span></div>
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
                  Create
                </button>
              </div>
            </div>
          </form>
        </Card>
      </div>
    </div>
  );
}
