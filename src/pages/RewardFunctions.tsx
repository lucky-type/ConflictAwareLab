import { useEffect, useState } from 'react';
import { PenSquare, PencilLine, ShieldCheck, Trash2 } from 'lucide-react';
import { Card, CardHeader } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import NewRewardFunctionModal, {
  RewardFunctionFormValues,
} from '../components/modals/NewRewardFunctionModal';
import { api, type RewardFunction as APIRewardFunction } from '../services/api';
import { useTopBar } from '../context/TopBarContext';

const formulaText = 'R = w_progress * distance - w_time * t - w_jerk * jerk';

// Helper to convert API format to UI format
const apiToFormValues = (rf: APIRewardFunction): RewardFunctionFormValues => ({
  name: rf.name,
  wProgress: rf.w_progress,
  wTime: rf.w_time,
  wJerk: rf.w_jerk,
  arrivalReward: rf.success_reward,
  crashPenalty: rf.crash_reward,
});

type ModalState = { open: boolean; editing: APIRewardFunction | null };

export default function RewardFunctions() {
  const [rewardFunctions, setRewardFunctions] = useState<APIRewardFunction[]>([]);
  const [modalState, setModalState] = useState<ModalState>({ open: false, editing: null });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { setActions } = useTopBar();

  useEffect(() => {
    loadRewardFunctions();
  }, []);

  useEffect(() => {
    setActions(
      <button
        onClick={() => setModalState({ open: true, editing: null })}
        className="flex items-center gap-2 rounded-md bg-notion-blue px-4 py-2 text-sm font-medium text-white hover:opacity-90 transition-opacity"
      >
        <PenSquare size={16} /> Add Function
      </button>
    );
    return () => setActions(null);
  }, []);

  const loadRewardFunctions = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getRewardFunctions();
      setRewardFunctions(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load reward functions');
      console.error('Failed to load reward functions:', err);
    } finally {
      setLoading(false);
    }
  };

  const closeModal = () => setModalState({ open: false, editing: null });

  const handleSave = async (values: RewardFunctionFormValues) => {
    try {
      setError(null);
      const apiData = {
        name: values.name,
        w_progress: Number(values.wProgress),
        w_time: Number(values.wTime),
        w_jerk: Number(values.wJerk),
        success_reward: Number(values.arrivalReward),
        crash_reward: Number(values.crashPenalty),
      };

      if (modalState.editing) {
        await api.updateRewardFunction(modalState.editing.id, apiData);
      } else {
        await api.createRewardFunction(apiData);
      }

      await loadRewardFunctions();
      closeModal();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save reward function';
      alert(`Error: ${message}`);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm('Delete this reward function? This action cannot be undone.')) return;

    try {
      setError(null);
      await api.deleteRewardFunction(id);
      await loadRewardFunctions();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete reward function';
      alert(`Error: ${message}`);
    }
  };

  const badgeTone = (isLocked: boolean) => (isLocked ? 'orange' : 'green');

  return (
    <div className="space-y-6">
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-notion-red">
          <strong>Error:</strong> {error}
        </div>
      )}

      {loading ? (
        <div className="text-center text-notion-text-secondary py-12">Loading reward functions...</div>
      ) : (
        <Card className="space-y-6">
          <CardHeader
            title="Definitions"
            actions={
              <div className="rounded-md border border-notion-border px-3 py-1 text-xs text-notion-text-secondary">
                {rewardFunctions.length} active
              </div>
            }
          />
          <p className="text-sm text-notion-text-secondary">
            Formula: <span className="font-mono text-notion-text">{formulaText}</span>. Each function can have custom
            weights and final rewards.
          </p>

          <div className="space-y-4">
            {rewardFunctions.length === 0 && (
              <div className="rounded-lg border border-dashed border-notion-border bg-notion-light-gray p-6 text-center text-sm text-notion-text-secondary">
                No reward functions yet. Add your first one to compare different reward schemes.
              </div>
            )}

            {rewardFunctions.map((reward) => (
              <div
                key={reward.id}
                className="space-y-4 rounded-lg border border-notion-border bg-notion-light-gray p-4"
              >
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div>
                    <p className="text-base font-semibold text-notion-text">{reward.name}</p>
                    <p className="text-xs text-notion-text-secondary">Created: {new Date(reward.created_at).toLocaleDateString()}</p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <Badge label={reward.is_locked ? 'Locked' : 'Available'} tone={badgeTone(reward.is_locked)} />
                    {!reward.is_locked && (
                      <div className="flex items-center gap-2 rounded-md border border-notion-border bg-green-50 px-3 py-1 text-notion-green">
                        <ShieldCheck size={14} /> Safe rollout
                      </div>
                    )}
                    <button
                      type="button"
                      onClick={() => setModalState({ open: true, editing: reward })}
                      className="flex items-center gap-1 rounded-md border border-notion-border px-3 py-1 text-notion-text-secondary hover:bg-notion-hover transition-colors"
                      disabled={reward.is_locked}
                    >
                      <PencilLine size={14} /> Edit
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDelete(reward.id)}
                      className="flex items-center gap-1 rounded-md border border-notion-border px-3 py-1 text-notion-red hover:bg-red-50 transition-colors"
                      disabled={reward.is_locked}
                    >
                      <Trash2 size={14} /> Delete
                    </button>
                  </div>
                </div>

                <div className="grid gap-3 text-sm text-notion-text md:grid-cols-4">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-notion-text-secondary">w_progress</p>
                    <p className="font-mono text-lg text-notion-text">{reward.w_progress}</p>
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-wide text-notion-text-secondary">w_time</p>
                    <p className="font-mono text-lg text-notion-text">{reward.w_time}</p>
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-wide text-notion-text-secondary">w_jerk</p>
                    <p className="font-mono text-lg text-notion-text">{reward.w_jerk}</p>
                  </div>
                  <div className="md:col-span-1">
                    <p className="text-xs uppercase tracking-wide text-notion-text-secondary">Formula</p>
                    <p className="font-mono text-xs text-notion-text">
                      R = {reward.w_progress} * d - {reward.w_time} * t - {reward.w_jerk} * j
                    </p>
                  </div>
                </div>

                <div className="grid gap-3 text-xs md:grid-cols-2">
                  <div className="rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-notion-green">
                    TARGET REACHED{' '}
                    <span className="text-notion-text font-semibold">
                      {reward.success_reward > 0 ? `+${reward.success_reward}` : reward.success_reward}
                    </span>
                  </div>
                  <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-notion-red">
                    CRASH{' '}
                    <span className="text-notion-text font-semibold">
                      {reward.crash_reward > 0 ? `+${reward.crash_reward}` : reward.crash_reward}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      <NewRewardFunctionModal
        open={modalState.open}
        onClose={closeModal}
        onSubmit={handleSave}
        initialValues={modalState.editing ? apiToFormValues(modalState.editing) : undefined}
      />
    </div>
  );
}
