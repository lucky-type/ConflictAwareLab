import { FormEvent, useState } from 'react';
import { X } from 'lucide-react';
import { Card, CardHeader } from '../common/Card';

interface Props {
  open: boolean;
  onClose: () => void;
}

const environments = ['Corridor_Standard_v1', 'Cityblock Flight v2', 'Warehouse Picker'];
const algorithms = ['PPO Baseline', 'PPO-LSTM', 'SAC Hybrid'];
const rewardFunctions = ['Standard Racing', 'Collision Avoidance'];
const models = ['Exp_204_model_final', 'navmesh-v2.4', 'precision-drop-v3'];

export default function NewExperimentModal({ open, onClose }: Props) {
  const [tab, setTab] = useState<'training' | 'simulation'>('training');
  const [experimentName, setExperimentName] = useState('Corridor_Run_01');
  const [totalSteps, setTotalSteps] = useState(1000000);
  const [snapshotInterval, setSnapshotInterval] = useState(50000);
  const [algorithm, setAlgorithm] = useState('PPO Baseline');
  const [rewardFunction, setRewardFunction] = useState('Standard Racing');
  const [environment, setEnvironment] = useState('Corridor_Standard_v1');
  const [episodes, setEpisodes] = useState(10);
  const [trainedModel, setTrainedModel] = useState('Exp_204_model_final');

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
            <CardHeader title="New Experiment" />
            <button onClick={onClose} className="text-slate-400 hover:text-white">
              <X size={20} />
            </button>
          </div>

          <div className="mb-4 grid grid-cols-2 rounded-xl border border-slate-800 bg-slate-900/80 text-sm font-semibold text-slate-300">
            <button
              className={`rounded-xl px-4 py-2 transition ${
                tab === 'training' ? 'bg-slate-800 text-white shadow-inner shadow-cyan-500/20' : 'hover:text-white'
              }`}
              onClick={() => setTab('training')}
              type="button"
            >
              Training
            </button>
            <button
              className={`rounded-xl px-4 py-2 transition ${
                tab === 'simulation' ? 'bg-slate-800 text-white shadow-inner shadow-cyan-500/20' : 'hover:text-white'
              }`}
              onClick={() => setTab('simulation')}
              type="button"
            >
              Simulation
            </button>
          </div>

          <form onSubmit={submit} className="grid gap-6 md:grid-cols-2">
            <div className="space-y-4">
              <label className="block space-y-2 text-sm">
                <span className="text-slate-300">Experiment name</span>
                <input
                  value={experimentName}
                  onChange={(e) => setExperimentName(e.target.value)}
                  className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  placeholder="e.g. PPO_Test_Run_01"
                />
              </label>

              {tab === 'training' ? (
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <label className="space-y-2">
                    <span className="text-slate-300">Total steps</span>
                    <input
                      type="number"
                      value={totalSteps}
                      onChange={(e) => setTotalSteps(parseInt(e.target.value) || 0)}
                      className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                    />
                  </label>
                  <label className="space-y-2">
                    <span className="text-slate-300">Snapshot interval</span>
                    <input
                      type="number"
                      value={snapshotInterval}
                      onChange={(e) => setSnapshotInterval(parseInt(e.target.value) || 0)}
                      className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                    />
                  </label>
                </div>
              ) : (
                <label className="block space-y-2 text-sm">
                  <span className="text-slate-300">Episodes</span>
                  <input
                    type="number"
                    value={episodes}
                    onChange={(e) => setEpisodes(parseInt(e.target.value) || 0)}
                    className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  />
                </label>
              )}

              {tab === 'simulation' && (
                <label className="block space-y-2 text-sm">
                  <span className="text-slate-300">Trained model (snapshot)</span>
                  <select
                    value={trainedModel}
                    onChange={(e) => setTrainedModel(e.target.value)}
                    className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  >
                    {models.map((model) => (
                      <option key={model}>{model}</option>
                    ))}
                  </select>
                </label>
              )}

              <label className="block space-y-2 text-sm">
                <span className="text-slate-300">Environment</span>
                <select
                  value={environment}
                  onChange={(e) => setEnvironment(e.target.value)}
                  className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                >
                  {environments.map((env) => (
                    <option key={env}>{env}</option>
                  ))}
                </select>
              </label>
            </div>

            <div className="space-y-4">
              {tab === 'training' ? (
                <label className="block space-y-2 text-sm">
                  <span className="text-slate-300">Algorithm</span>
                  <select
                    value={algorithm}
                    onChange={(e) => setAlgorithm(e.target.value)}
                    className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  >
                    {algorithms.map((algo) => (
                      <option key={algo}>{algo}</option>
                    ))}
                  </select>
                </label>
              ) : (
                <label className="block space-y-2 text-sm">
                  <span className="text-slate-300">Reward function</span>
                  <select
                    value={rewardFunction}
                    onChange={(e) => setRewardFunction(e.target.value)}
                    className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  >
                    {rewardFunctions.map((reward) => (
                      <option key={reward}>{reward}</option>
                    ))}
                  </select>
                </label>
              )}

              {tab === 'simulation' && (
                <label className="block space-y-2 text-sm">
                  <span className="text-slate-300">Environment (override)</span>
                  <select
                    value={environment}
                    onChange={(e) => setEnvironment(e.target.value)}
                    className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  >
                    {environments.map((env) => (
                      <option key={env}>{env}</option>
                    ))}
                  </select>
                </label>
              )}

              {tab === 'training' && (
                <label className="block space-y-2 text-sm">
                  <span className="text-slate-300">Reward function</span>
                  <select
                    value={rewardFunction}
                    onChange={(e) => setRewardFunction(e.target.value)}
                    className="w-full rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-slate-100 focus:border-cyan-400"
                  >
                    {rewardFunctions.map((reward) => (
                      <option key={reward}>{reward}</option>
                    ))}
                  </select>
                </label>
              )}

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
                  Start
                </button>
              </div>
            </div>
          </form>
        </Card>
      </div>
    </div>
  );
}
