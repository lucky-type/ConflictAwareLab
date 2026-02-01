// import { X, CheckCircle2, Clock, PlayCircle, PauseCircle, AlertCircle } from 'lucide-react';
// import { Curriculum } from '../../services/api';
// import { Badge } from '../common/Badge';

// type Props = {
//     curriculum: Curriculum | null;
//     onClose: () => void;
// };

// export default function CurriculumDetailsModal({ curriculum, onClose }: Props) {
//     if (!curriculum) return null;

//     const totalSteps = curriculum.steps.length;

//     return (
//         <div 
//             className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm"
//             onClick={onClose}
//         >
//             <div 
//                 className="flex h-[80vh] w-full max-w-4xl flex-col rounded-2xl border border-notion-border bg-slate-950 shadow-2xl"
//                 onClick={(e) => e.stopPropagation()}
//             >
//                 {/* Header */}
//                 <div className="flex items-center justify-between border-b border-notion-border p-6">
//                     <div>
//                         <div className="flex items-center gap-3">
//                             <h2 className="text-xl font-bold text-notion-text">{curriculum.name}</h2>
//                             <Badge label={curriculum.status} tone={getStatusTone(curriculum.status)} />
//                         </div>
//                         {curriculum.description && (
//                             <p className="mt-1 text-sm text-notion-text-secondary">{curriculum.description}</p>
//                         )}
//                     </div>
//                     <button onClick={onClose} className="rounded-lg p-2 text-notion-text-secondary hover:bg-slate-900 hover:text-notion-text">
//                         <X size={20} />
//                     </button>
//                 </div>

//                 {/* Body */}
//                 <div className="flex-1 overflow-y-auto p-8">
//                     <div className="space-y-8">
//                         <div className="space-y-4">
//                             <h3 className="text-lg font-semibold text-notion-text">Pipeline Steps</h3>
//                             <div className="relative space-y-8 pl-8 before:absolute before:left-3.5 before:top-4 before:h-full before:w-0.5 before:bg-notion-hover">
//                                 {curriculum.steps.map((step, idx) => {
//                                     const isActive = curriculum.status === 'Running' && step.order === curriculum.current_step_order;
//                                     const isCompleted = step.is_completed;
//                                     const isPending = !isCompleted && !isActive;

//                                     return (
//                                         <div key={step.id} className="relative">
//                                             {/* Status Dot */}
//                                             <div className={`absolute -left-[35px] flex h-8 w-8 items-center justify-center rounded-full border-2 border-slate-950 shadow-sm ${isCompleted ? 'bg-emerald-500 text-white' :
//                                                     isActive ? 'bg-cyan-500 text-white animate-pulse' :
//                                                         'bg-notion-hover text-notion-text-secondary'
//                                                 }`}>
//                                                 {isCompleted ? <CheckCircle2 size={16} /> :
//                                                     isActive ? <PlayCircle size={16} /> :
//                                                         <span className="text-xs font-bold">{step.order}</span>}
//                                             </div>

//                                             <div className={`rounded-xl border p-4 transition ${isActive ? 'border-cyan-500/50 bg-cyan-500/5' :
//                                                     isCompleted ? 'border-emerald-500/20 bg-emerald-500/5' :
//                                                         'border-notion-border bg-slate-900/30'
//                                                 }`}>
//                                                 <div className="mb-3 flex items-start justify-between">
//                                                     <div>
//                                                         <div className="flex items-center gap-2">
//                                                             <h4 className="font-semibold text-notion-text">
//                                                                 Step {step.order}: {formatStepType(step.step_type)}
//                                                             </h4>
//                                                             {isActive && (
//                                                                 <span className="flex items-center gap-1 rounded-full bg-cyan-500/20 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-notion-blue">
//                                                                     <Clock size={10} /> Active
//                                                                 </span>
//                                                             )}
//                                                         </div>
//                                                         <p className="text-xs text-notion-text-secondary">
//                                                             Target Success: {(step.target_success_rate * 100).toFixed(0)}% • Min Episodes: {step.min_episodes}
//                                                         </p>
//                                                     </div>

//                                                     {/* Config Summary Badge */}
//                                                     <div className="flex gap-2">
//                                                         <span className="rounded bg-notion-hover px-2 py-1 text-xs text-notion-text">
//                                                             Total Steps: {step.config.total_steps.toLocaleString()}
//                                                         </span>
//                                                     </div>
//                                                 </div>

//                                                 {/* Details Table */}
//                                                 <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-xs text-notion-text-secondary sm:grid-cols-3">
//                                                     <div>
//                                                         <span className="block text-notion-text-tertiary">Environment ID</span>
//                                                         <span className="text-notion-text">{step.config.env_id}</span>
//                                                     </div>
//                                                     <div>
//                                                         <span className="block text-notion-text-tertiary">Agent/Connector ID</span>
//                                                         <span className="text-notion-text">
//                                                             {step.step_type === 'residual'
//                                                                 ? step.config.residual_connector_id
//                                                                 : step.config.agent_id}
//                                                         </span>
//                                                     </div>
//                                                     <div>
//                                                         <span className="block text-notion-text-tertiary">Reward Function ID</span>
//                                                         <span className="text-notion-text">{step.config.reward_id}</span>
//                                                     </div>
//                                                     {step.safety_config && step.safety_config.enabled && (
//                                                         <div className="col-span-full mt-2 rounded border border-rose-500/20 bg-rose-500/5 p-2">
//                                                             <div className="flex items-center gap-2 text-rose-600 font-medium mb-1">
//                                                                 <AlertCircle size={12} /> Safety Constraints Enabled
//                                                             </div>
//                                                             <div className="grid grid-cols-2 gap-4 text-xs">
//                                                                 <span>Risk Budget: {step.safety_config.risk_budget}</span>
//                                                                 <span>Initial Lambda: {step.safety_config.initial_lambda}</span>
//                                                                 <span>Collision Weight: {step.safety_config.collision_weight}</span>
//                                                                 <span>Near-Miss Weight: {step.safety_config.near_miss_weight}</span>
//                                                                 <span>Danger Zone Weight: {step.safety_config.danger_zone_weight}</span>
//                                                                 <span>Ignore Walls: {step.safety_config.ignore_walls ? 'Yes' : 'No'}</span>
//                                                             </div>
//                                                         </div>
//                                                     )}
//                                                 </div>
//                                             </div>
//                                         </div>
//                                     );
//                                 })}
//                             </div>
//                         </div>
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// }

// function getStatusTone(status: string) {
//     switch (status) {
//         case 'Running': return 'bg-emerald-500/20 text-emerald-600 border-emerald-500/40';
//         case 'Paused': return 'bg-sky-500/20 text-sky-600 border-sky-500/40';
//         case 'Completed': return 'bg-cyan-500/20 text-cyan-600 border-cyan-500/40';
//         case 'Failed': return 'bg-rose-500/20 text-rose-600 border-rose-500/40';
//         default: return 'bg-slate-500/20 text-notion-text border-slate-500/40';
//     }
// }

// function formatStepType(type: string) {
//     return type.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
// }
