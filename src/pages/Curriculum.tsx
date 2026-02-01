// import { useEffect, useState, useCallback } from 'react';
// import { CheckCircle2, GraduationCap, Loader2, PauseCircle, Play, Plus, Trash2, XCircle, AlertCircle } from 'lucide-react';
// import { Badge } from '../components/common/Badge';
// import { Card, CardHeader } from '../components/common/Card';
// import { api, Curriculum, CurriculumStatus } from '../services/api';
// import { useTopBar } from '../context/TopBarContext';
// import CurriculumModal from '../components/modals/CurriculumModal';
// import CurriculumDetailsModal from '../components/modals/CurriculumDetailsModal';

// const statusVisuals: Record<CurriculumStatus, { icon: typeof Loader2; accent: string; pulse?: boolean }> = {
//     Pending: { icon: PauseCircle, accent: 'text-notion-text' },
//     Running: { icon: Loader2, accent: 'text-emerald-600', pulse: true },
//     Paused: { icon: PauseCircle, accent: 'text-sky-600' },
//     Completed: { icon: CheckCircle2, accent: 'text-cyan-600' },
//     Failed: { icon: XCircle, accent: 'text-rose-600' },
// };

// const statusBadgeTone: Record<CurriculumStatus, string> = {
//     Pending: 'bg-slate-500/20 text-notion-text border-slate-500/40',
//     Running: 'bg-emerald-500/20 text-emerald-600 border-emerald-500/40',
//     Paused: 'bg-sky-500/20 text-sky-600 border-sky-500/40',
//     Completed: 'bg-cyan-500/20 text-cyan-600 border-cyan-500/40',
//     Failed: 'bg-rose-500/20 text-rose-600 border-rose-500/40',
// };

// export default function CurriculumPage() {
//     const { setActions } = useTopBar();
//     const [curriculums, setCurriculums] = useState<Curriculum[]>([]);
//     const [loading, setLoading] = useState(true);
//     const [error, setError] = useState<string | null>(null);
//     const [isModalOpen, setIsModalOpen] = useState(false);
//     const [selectedCurriculum, setSelectedCurriculum] = useState<Curriculum | null>(null);

//     const loadCurriculums = useCallback(async () => {
//         try {
//             setLoading(true);
//             setError(null);
//             const data = await api.getCurriculums();
//             setCurriculums(data);
//         } catch (err) {
//             setError(err instanceof Error ? err.message : 'Failed to load curriculums');
//             console.error('Failed to load curriculums:', err);
//         } finally {
//             setLoading(false);
//         }
//     }, []);

//     useEffect(() => {
//         loadCurriculums();
//     }, [loadCurriculums]);

//     // Update selected curriculum details when curriculums refresh
//     useEffect(() => {
//         if (selectedCurriculum) {
//             const updated = curriculums.find(c => c.id === selectedCurriculum.id);
//             if (updated && JSON.stringify(updated) !== JSON.stringify(selectedCurriculum)) {
//                 setSelectedCurriculum(updated);
//             }
//         }
//     }, [curriculums, selectedCurriculum]);

//     // Auto-refresh periodically if any are running
//     useEffect(() => {
//         if (!curriculums.some(c => c.status === 'Running')) return;
//         const interval = setInterval(loadCurriculums, 5000);
//         return () => clearInterval(interval);
//     }, [curriculums, loadCurriculums]);

//     useEffect(() => {
//         setActions(
//             <button
//                 onClick={() => setIsModalOpen(true)}
//                 className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-notion-blue to-notion-blue px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-notion-blue/20"
//             >
//                 <Plus size={16} /> Create
//             </button>
//         );
//         return () => setActions(null);
//     }, [setActions]);

//     const handleStart = async (id: number) => {
//         try {
//             await api.startCurriculum(id);
//             await loadCurriculums();
//         } catch (err) {
//             alert('Failed to start curriculum: ' + (err instanceof Error ? err.message : 'Unknown error'));
//         }
//     };

//     const handlePause = async (id: number) => {
//         try {
//             await api.pauseCurriculum(id);
//             await loadCurriculums();
//         } catch (err) {
//             alert('Failed to pause curriculum: ' + (err instanceof Error ? err.message : 'Unknown error'));
//         }
//     };

//     const handleDelete = async (id: number) => {
//         if (!confirm('Are you sure you want to delete this curriculum?')) return;

//         try {
//             await api.deleteCurriculum(id);
//             await loadCurriculums();
//             if (selectedCurriculum?.id === id) setSelectedCurriculum(null);
//         } catch (err) {
//             alert('Failed to delete curriculum: ' + (err instanceof Error ? err.message : 'Unknown error'));
//         }
//     };

//     if (loading && curriculums.length === 0) {
//         return (
//             <div className="flex h-64 items-center justify-center">
//                 <Loader2 className="h-8 w-8 animate-spin text-notion-blue" />
//             </div>
//         );
//     }

//     if (error) {
//         return (
//             <Card className="p-6">
//                 <div className="flex items-center gap-3 text-rose-400">
//                     <AlertCircle size={20} />
//                     <p>{error}</p>
//                 </div>
//             </Card>
//         );
//     }

//     return (
//         <div className="space-y-6">
//             <Card className="space-y-4">
//                 <CardHeader
//                     title="Curriculum Pipelines"
//                     subtitle="Multi-stage training pipelines with automatic model linking"
//                 />

//                 {curriculums.length === 0 ? (
//                     <p className="rounded-xl border border-dashed border-notion-border bg-notion-light-gray px-4 py-8 text-center text-sm text-notion-text-secondary">
//                         No curriculum pipelines yet. Create one to define sequential training stages.
//                     </p>
//                 ) : (
//                     <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
//                         {curriculums.map((curriculum) => (
//                             <CurriculumCard
//                                 key={curriculum.id}
//                                 curriculum={curriculum}
//                                 onStart={() => handleStart(curriculum.id)}
//                                 onPause={() => handlePause(curriculum.id)}
//                                 onDelete={() => handleDelete(curriculum.id)}
//                                 onClick={() => setSelectedCurriculum(curriculum)}
//                             />
//                         ))}
//                     </div>
//                 )}
//             </Card>

//             <CurriculumModal
//                 open={isModalOpen}
//                 onClose={() => setIsModalOpen(false)}
//                 onSave={loadCurriculums}
//             />

//             <CurriculumDetailsModal
//                 curriculum={selectedCurriculum}
//                 onClose={() => setSelectedCurriculum(null)}
//             />
//         </div>
//     );
// }

// type CurriculumCardProps = {
//     curriculum: Curriculum;
//     onStart: () => void;
//     onPause: () => void;
//     onDelete: () => void;
//     onClick: () => void;
// };

// function CurriculumCard({ curriculum, onStart, onPause, onDelete, onClick }: CurriculumCardProps) {
//     const statusInfo = statusVisuals[curriculum.status];
//     const StatusIcon = statusInfo.icon;

//     const totalSteps = curriculum.steps.length;
//     const completedSteps = curriculum.steps.filter(s => s.is_completed).length;
//     const progressPercent = totalSteps > 0 ? Math.round((completedSteps / totalSteps) * 100) : 0;

//     return (
//         <div
//             className="flex h-full flex-col rounded-2xl border-2 border-slate-900 bg-notion-light-gray p-4 transition cursor-pointer hover:bg-slate-900/80 hover:border-cyan-500/40"
//             onClick={onClick}
//         >
//             <div className="flex items-start justify-between">
//                 <div className="space-y-1">
//                     <span className="inline-flex items-center gap-1 rounded-full border border-notion-border px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-notion-text">
//                         <GraduationCap size={12} /> Pipeline
//                     </span>
//                     <h4 className="text-base font-semibold text-notion-text">{curriculum.name}</h4>
//                     {curriculum.description && (
//                         <p className="text-xs text-notion-text-secondary">{curriculum.description}</p>
//                     )}
//                 </div>
//                 <Badge label={curriculum.status} tone={statusBadgeTone[curriculum.status]} />
//             </div>

//             <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-notion-text-secondary">
//                 <div className={`flex items-center gap-1 text-sm ${statusInfo.accent} ${statusInfo.pulse ? 'animate-pulse' : ''}`}>
//                     <StatusIcon size={16} />
//                     {curriculum.status}
//                 </div>
//                 <span>•</span>
//                 <span>Step {curriculum.current_step_order}/{totalSteps}</span>
//             </div>

//             <div className="mt-4 flex flex-col gap-2 text-sm text-notion-text">
//                 <div className="flex justify-between text-xs text-notion-text-secondary">
//                     <span>Progress</span>
//                     <span>{progressPercent}%</span>
//                 </div>
//                 <div className="h-1.5 rounded-full bg-notion-hover">
//                     <div className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-cyan-500" style={{ width: `${progressPercent}%` }} />
//                 </div>
//             </div>

//             <div className="mt-auto flex gap-2 pt-4" onClick={(e) => e.stopPropagation()}>
//                 {curriculum.status === 'Pending' && (
//                     <button
//                         onClick={onStart}
//                         className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-emerald-200 px-3 py-2 text-sm text-emerald-600 hover:border-emerald-500/60"
//                         title="Start Curriculum"
//                     >
//                         <Play size={14} /> Start
//                     </button>
//                 )}
//                 {curriculum.status === 'Running' && (
//                     <button
//                         onClick={onPause}
//                         className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-sky-200 px-3 py-2 text-sm text-sky-600 hover:border-sky-500/60"
//                         title="Pause Curriculum"
//                     >
//                         <PauseCircle size={14} /> Pause
//                     </button>
//                 )}
//                 {curriculum.status === 'Paused' && (
//                     <button
//                         onClick={onStart}
//                         className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-emerald-200 px-3 py-2 text-sm text-emerald-600 hover:border-emerald-500/60"
//                         title="Resume Curriculum"
//                     >
//                         <Play size={14} /> Resume
//                     </button>
//                 )}
//                 {(curriculum.status === 'Completed' || curriculum.status === 'Failed' || curriculum.status === 'Pending' || curriculum.status === 'Paused') && (
//                     <button
//                         onClick={onDelete}
//                         className="flex items-center justify-center rounded-xl border border-rose-200 px-3 py-2 text-sm text-rose-600 hover:border-rose-500/60"
//                         title="Delete Curriculum"
//                     >
//                         <Trash2 size={14} />
//                     </button>
//                 )}
//             </div>
//         </div>
//     );
// }
