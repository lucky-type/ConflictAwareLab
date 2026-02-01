import { FormEvent, useEffect, useMemo, useState } from 'react';
import { X } from 'lucide-react';
import { Card, CardHeader } from '../common/Card';

export interface RewardFunctionFormValues {
  name: string;
  wProgress: number | '';
  wTime: number | '';
  wJerk: number | '';
  arrivalReward: number | '';
  crashPenalty: number | '';
}

interface Props {
  open: boolean;
  onClose: () => void;
  onSubmit: (values: RewardFunctionFormValues) => void;
  initialValues?: RewardFunctionFormValues;
}

type ValidationErrors = Partial<Record<keyof RewardFunctionFormValues, string>>;

const validate = (values: RewardFunctionFormValues): ValidationErrors => {
  const errors: ValidationErrors = {};

  if (!values.name.trim()) errors.name = 'Required';
  const wProgress = typeof values.wProgress === 'number' ? values.wProgress : parseFloat(String(values.wProgress));
  const wTime = typeof values.wTime === 'number' ? values.wTime : parseFloat(String(values.wTime));
  const wJerk = typeof values.wJerk === 'number' ? values.wJerk : parseFloat(String(values.wJerk));
  const arrivalReward = typeof values.arrivalReward === 'number' ? values.arrivalReward : parseFloat(String(values.arrivalReward));
  const crashPenalty = typeof values.crashPenalty === 'number' ? values.crashPenalty : parseFloat(String(values.crashPenalty));

  if (isNaN(wProgress) || wProgress <= 0) errors.wProgress = 'Must be > 0';
  if (isNaN(wTime) || wTime < 0) errors.wTime = 'Must be \u2265 0';
  if (isNaN(wJerk) || wJerk < 0) errors.wJerk = 'Must be \u2265 0';
  if (isNaN(arrivalReward) || arrivalReward <= 0) errors.arrivalReward = 'Must be positive';
  if (isNaN(crashPenalty) || crashPenalty >= 0) errors.crashPenalty = 'Must be negative';

  return errors;
};

const defaultValues: RewardFunctionFormValues = {
  name: 'Standard Racing',
  wProgress: 2,
  wTime: 1,
  wJerk: 1,
  arrivalReward: 1000,
  crashPenalty: -1000,
};

export default function NewRewardFunctionModal({ open, onClose, onSubmit, initialValues }: Props) {
  const [values, setValues] = useState<RewardFunctionFormValues>(initialValues ?? defaultValues);

  useEffect(() => {
    if (open) {
      setValues(initialValues ?? defaultValues);
    }
  }, [open, initialValues]);

  const isEditMode = Boolean(initialValues);
  const errors = useMemo(() => validate(values), [values]);
  const isValid = Object.keys(errors).length === 0;
  const rewardPreview = useMemo(
    () => {
      const arrival = typeof values.arrivalReward === 'number' ? values.arrivalReward : 0;
      const crash = typeof values.crashPenalty === 'number' ? values.crashPenalty : 0;
      return {
        arrival: arrival > 0 ? `+${arrival}` : `${arrival}`,
        crash: crash > 0 ? `+${crash}` : `${crash}`,
      };
    },
    [values.arrivalReward, values.crashPenalty]
  );

  if (!open) return null;

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!isValid) return;

    // Convert empty strings to numbers before submitting
    const sanitizedValues: RewardFunctionFormValues = {
      name: values.name,
      wProgress: typeof values.wProgress === 'number' ? values.wProgress : parseFloat(String(values.wProgress)) || 2,
      wTime: typeof values.wTime === 'number' ? values.wTime : parseFloat(String(values.wTime)) || 1,
      wJerk: typeof values.wJerk === 'number' ? values.wJerk : parseFloat(String(values.wJerk)) || 1,
      arrivalReward: typeof values.arrivalReward === 'number' ? values.arrivalReward : parseFloat(String(values.arrivalReward)) || 1000,
      crashPenalty: typeof values.crashPenalty === 'number' ? values.crashPenalty : parseFloat(String(values.crashPenalty)) || -1000,
    };

    onSubmit(sanitizedValues);
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm"
    >
      <div className="w-full max-w-4xl overflow-y-auto max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
        <Card className="p-6">
          <div className="flex items-start justify-between">
            <CardHeader title={isEditMode ? 'Update Reward Function' : 'New Reward Function'} />
            <button onClick={onClose} className="text-notion-text-secondary hover:text-notion-text">
              <X size={20} />
            </button>
          </div>

          <form onSubmit={submit} className="grid gap-6 md:grid-cols-2">
            <div className="space-y-4">
              <div className="rounded-xl border border-notion-border bg-notion-light-gray p-4 text-sm text-notion-text">
                <p className="text-xs uppercase tracking-wide text-notion-blue">Reward formula</p>
                <p className="font-mono text-xs text-notion-text">
                  R = w_progress * distance - w_time * t - w_jerk * jerk
                </p>
                <p className="mt-2 text-notion-text-secondary">
                  Configure weights for progress control, time penalty, and jerk smoothing.
                </p>
              </div>

              <label className="block space-y-2 text-sm">
                <div className="flex items-center justify-between text-notion-text">
                  <span>Function name</span>
                  {errors.name && <span className="text-xs text-rose-400">{errors.name}</span>}
                </div>
                <input
                  value={values.name}
                  onChange={(e) => setValues((prev) => ({ ...prev, name: e.target.value }))}
                  className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                  placeholder="e.g. Agile Corridor"
                  aria-invalid={Boolean(errors.name)}
                />
              </label>
            </div>

            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-3 text-sm">
                <label className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-notion-text">w_progress</span>
                    {errors.wProgress && <span className="text-xs text-rose-400">{errors.wProgress}</span>}
                  </div>
                  <input
                    type="number"
                    step="any"
                    min="-10"
                    max="10"
                    value={values.wProgress}
                    onChange={(e) =>
                      setValues((prev) => ({ ...prev, wProgress: e.target.value === '' ? '' : parseFloat(e.target.value) }))
                    }
                    className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                    aria-invalid={Boolean(errors.wProgress)}
                  />
                </label>
                <label className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-notion-text">w_time</span>
                    {errors.wTime && <span className="text-xs text-rose-400">{errors.wTime}</span>}
                  </div>
                  <input
                    type="number"
                    step="any"
                    min="-10"
                    max="10"
                    value={values.wTime}
                    onChange={(e) =>
                      setValues((prev) => ({ ...prev, wTime: e.target.value === '' ? '' : parseFloat(e.target.value) }))
                    }
                    className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                    aria-invalid={Boolean(errors.wTime)}
                  />
                </label>
                <label className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-notion-text">w_jerk</span>
                    {errors.wJerk && <span className="text-xs text-rose-400">{errors.wJerk}</span>}
                  </div>
                  <input
                    type="number"
                    step="any"
                    min="-10"
                    max="10"
                    value={values.wJerk}
                    onChange={(e) =>
                      setValues((prev) => ({ ...prev, wJerk: e.target.value === '' ? '' : parseFloat(e.target.value) }))
                    }
                    className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                    aria-invalid={Boolean(errors.wJerk)}
                  />
                </label>
              </div>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <label className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-notion-text">Reward (target reached)</span>
                    {errors.arrivalReward && <span className="text-xs text-rose-400">{errors.arrivalReward}</span>}
                  </div>
                  <input
                    type="number"
                    min="-1000"
                    max="1000"
                    value={values.arrivalReward}
                    onChange={(e) =>
                      setValues((prev) => ({ ...prev, arrivalReward: e.target.value === '' ? '' : parseFloat(e.target.value) }))
                    }
                    className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                    aria-invalid={Boolean(errors.arrivalReward)}
                  />
                </label>
                <label className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-notion-text">Penalty (crash)</span>
                    {errors.crashPenalty && <span className="text-xs text-rose-400">{errors.crashPenalty}</span>}
                  </div>
                  <input
                    type="number"
                    min="-1000"
                    max="1000"
                    value={values.crashPenalty}
                    onChange={(e) =>
                      setValues((prev) => ({ ...prev, crashPenalty: e.target.value === '' ? '' : parseFloat(e.target.value) }))
                    }
                    className="w-full rounded-xl border border-notion-border bg-white px-3 py-2 text-notion-text focus:border-notion-blue focus:ring-1 focus:ring-notion-blue focus:outline-none"
                    aria-invalid={Boolean(errors.crashPenalty)}
                  />
                </label>
              </div>

              <div className="flex items-center gap-3 rounded-xl border border-notion-border bg-notion-light-gray p-3 text-xs">
                <div className="flex-1 rounded-lg bg-emerald-100 border border-emerald-200 px-3 py-2 text-emerald-700">
                  TARGET REACHED <span className="font-semibold">{rewardPreview.arrival}</span>
                </div>
                <div className="flex-1 rounded-lg bg-rose-100 border border-rose-200 px-3 py-2 text-rose-700">
                  CRASH <span className="font-semibold">{rewardPreview.crash}</span>
                </div>
              </div>

              <div className="flex items-center justify-end gap-3">
                <button
                  type="button"
                  onClick={onClose}
                  className="rounded-xl border border-notion-border px-4 py-2 text-sm text-notion-text hover:bg-notion-hover"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!isValid}
                  className="rounded-xl bg-notion-blue px-4 py-2 text-sm font-semibold text-white hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {isEditMode ? 'Save changes' : 'Create'}
                </button>
              </div>
            </div>
          </form>
        </Card>
      </div>
    </div>
  );
}
