import { FormEvent, useState } from 'react';
import { X } from 'lucide-react';
import { Card, CardHeader } from '../common/Card';
import { quickActions } from '../../data/mockData';

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function NewExperimentModal({ open, onClose }: Props) {
  const [name, setName] = useState('');
  const [environment, setEnvironment] = useState('Cityblock Flight v2');
  const [algorithm, setAlgorithm] = useState('PPO-LSTM');
  const [priority, setPriority] = useState('standard');

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
            <CardHeader title="New Experiment" />
            <button onClick={onClose} className="text-slate-400 hover:text-white">
              <X size={20} />
            </button>
          </div>
          <form onSubmit={submit} className="grid gap-6 md:grid-cols-2">
            <div className="space-y-4">
              <label className="block space-y-2 text-sm">
                <span className="text-slate-300">Experiment name</span>
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  placeholder="Precision pick-and-place"
                />
              </label>
              <label className="block space-y-2 text-sm">
                <span className="text-slate-300">Environment</span>
                <select
                  value={environment}
                  onChange={(e) => setEnvironment(e.target.value)}
                  className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                >
                  <option>Cityblock Flight v2</option>
                  <option>Warehouse Picker</option>
                  <option>Maritime Dock Ops</option>
                  <option>Canyon Run</option>
                </select>
              </label>
              <label className="block space-y-2 text-sm">
                <span className="text-slate-300">Algorithm</span>
                <select
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value)}
                  className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                >
                  <option>PPO-LSTM</option>
                  <option>SAC Hybrid</option>
                  <option>DreamerV3</option>
                </select>
              </label>
            </div>
            <div className="space-y-4">
              <fieldset className="space-y-2 text-sm">
                <span className="text-slate-300">Priority</span>
                <div className="grid grid-cols-3 gap-2">
                  {['standard', 'high', 'urgent'].map((level) => (
                    <label
                      key={level}
                      className="flex cursor-pointer flex-col items-center gap-1 rounded-xl border border-slate-800/80 bg-slate-900/80 px-3 py-2 text-slate-200 hover:border-cyan-500/40"
                    >
                      <input
                        type="radio"
                        value={level}
                        checked={priority === level}
                        onChange={() => setPriority(level)}
                        className="accent-cyan-500"
                      />
                      <span className="capitalize">{level}</span>
                    </label>
                  ))}
                </div>
              </fieldset>
              <div>
                <p className="mb-2 text-sm text-slate-300">Suggested quick actions</p>
                <div className="grid grid-cols-3 gap-2">
                  {quickActions.map((action) => (
                    <button
                      type="button"
                      key={action.label}
                      className="flex items-center gap-2 rounded-xl border border-slate-800/70 bg-slate-900/80 px-3 py-2 text-xs text-slate-200 hover:border-cyan-500/40"
                    >
                      <action.icon size={16} className="text-cyan-400" />
                      <span>{action.label}</span>
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex items-center justify-end gap-3 pt-4">
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
                  Launch experiment
                </button>
              </div>
            </div>
          </form>
        </Card>
      </div>
    </div>
  );
}
