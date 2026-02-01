/*
import { useEffect, useState } from 'react';
import { X, Plus, Trash2, ArrowRight, ArrowLeft, Save, AlertCircle } from 'lucide-react';
// import { api, Agent, Environment, RewardFunction, ResidualConnector, CurriculumCreate, CurriculumStepCreate } from '../../services/api';

type Props = {
    open: boolean;
    onClose: () => void;
    onSave: () => void;
};

export default function CurriculumModal({ open, onClose, onSave }: Props) {
    const [step, setStep] = useState(1);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Form Data
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const [pipelineSteps, setPipelineSteps] = useState<any[]>([
        {
            order: 1,
            step_type: 'training',
            config: {
                env_id: 0,
                agent_id: 0,
                reward_id: 0,
                total_steps: 100000,
                snapshot_freq: 10000,
                max_ep_length: 2000
            },
            target_success_rate: 0.95,
            min_episodes: 50,
            safety_config: {
                enabled: false,
                risk_budget: 0.1,
                initial_lambda: 0.5,
                lambda_learning_rate: 0.1,
                update_frequency: 100,
                target_lambda: null,
                warmup_episodes: 0,
                warmup_schedule: 'exponential',
                cost_signal_preset: 'default',
                collision_weight: 0.1,
                near_miss_weight: 0.01,
                danger_zone_weight: 0.005,
                near_miss_threshold: 1.5,
                ignore_walls: true,
                wall_near_miss_weight: 0.005,
                wall_danger_zone_weight: 0.01,
                wall_near_miss_threshold: 1.0
            }
        }
    ]);

    // Data Sources
    const [environments, setEnvironments] = useState<any[]>([]);
    const [agents, setAgents] = useState<any[]>([]);
    const [rewards, setRewards] = useState<any[]>([]);
    const [connectors, setConnectors] = useState<any[]>([]);

    useEffect(() => {
        if (open) {
            loadDependencies();
            setStep(1);
            setName('');
            setDescription('');
            setPipelineSteps([{
                order: 1,
                step_type: 'training',
                config: {
                    env_id: 0,
                    agent_id: 0,
                    reward_id: 0,
                    total_steps: 100000,
                    snapshot_freq: 10000,
                    max_ep_length: 2000
                },
                target_success_rate: 0.95,
                min_episodes: 50,
                safety_config: { enabled: false, risk_budget: 0.02, initial_lambda: 0.0, lambda_learning_rate: 0.02, update_frequency: 50, target_lambda: null, warmup_episodes: 0, warmup_schedule: 'exponential', cost_signal_preset: 'balanced', collision_weight: 0.1, near_miss_weight: 0.01, danger_zone_weight: 0.005, near_miss_threshold: 1.5, ignore_walls: true, wall_near_miss_weight: 0.005, wall_danger_zone_weight: 0.01, wall_near_miss_threshold: 1.0 }
            }]);
            setError(null);
        }
    }, [open]);

    const loadDependencies = async () => {
        try {
            // commented out api calls
            // const [envs, agts, rwds, conns] = await Promise.all([
            //     api.getEnvironments(),
            //     api.getAgents(),
            //     api.getRewardFunctions(),
            //     api.getResidualConnectors()
            // ]);
            // setEnvironments(envs);
            // setAgents(agts);
            // setRewards(rwds);
            // setConnectors(conns);

            // Initialize first step with valid default IDs
            // setPipelineSteps(prev => {
            //     const newSteps = [...prev];
            //     if (newSteps[0]) {
            //         if (envs.length > 0) newSteps[0].config.env_id = envs[0].id;
            //         if (agts.length > 0) newSteps[0].config.agent_id = agts[0].id;
            //         if (rwds.length > 0) newSteps[0].config.reward_id = rwds[0].id;
            //     }
            //     return newSteps;
            // });

        } catch (err) {
            console.error('Failed to load dependencies:', err);
            setError('Failed to load form data. Please try closing and reopening.');
        }
    };

    const handleStepChange = (index: number, field: any, value: any) => {
        const newSteps = [...pipelineSteps];
        newSteps[index] = { ...newSteps[index], [field]: value };
        setPipelineSteps(newSteps);
    };

    const handleConfigChange = (index: number, field: string, value: any) => {
        const newSteps = [...pipelineSteps];
        newSteps[index] = {
            ...newSteps[index],
            config: { ...newSteps[index].config, [field]: value }
        };
        setPipelineSteps(newSteps);
    };

    const addPipelineStep = () => {
        const lastStep = pipelineSteps[pipelineSteps.length - 1];
        setPipelineSteps([
            ...pipelineSteps,
            {
                order: pipelineSteps.length + 1,
                step_type: 'fine_tuning',
                config: {
                    ...lastStep.config,
                    // Auto-inherit agent/env/reward from previous step initially
                },
                target_success_rate: 0.95,
                min_episodes: 50,
                safety_config: { 
                    enabled: lastStep.safety_config?.enabled ?? false,
                    risk_budget: lastStep.safety_config?.risk_budget ?? 0.02,
                    initial_lambda: lastStep.safety_config?.initial_lambda ?? 0.0,
                    lambda_learning_rate: lastStep.safety_config?.lambda_learning_rate ?? 0.02,
                    update_frequency: lastStep.safety_config?.update_frequency ?? 50,
                    cost_signal_preset: lastStep.safety_config?.cost_signal_preset ?? 'balanced',
                    collision_weight: lastStep.safety_config?.collision_weight ?? 0.1,
                    near_miss_weight: lastStep.safety_config?.near_miss_weight ?? 0.01,
                    danger_zone_weight: lastStep.safety_config?.danger_zone_weight ?? 0.005,
                    near_miss_threshold: lastStep.safety_config?.near_miss_threshold ?? 1.5,
                    ignore_walls: lastStep.safety_config?.ignore_walls ?? true,
                    wall_near_miss_weight: lastStep.safety_config?.wall_near_miss_weight ?? 0.005,
                    wall_danger_zone_weight: lastStep.safety_config?.wall_danger_zone_weight ?? 0.01,
                    wall_near_miss_threshold: lastStep.safety_config?.wall_near_miss_threshold ?? 1.0
                }
            }
        ]);
    };

    const removePipelineStep = (index: number) => {
        if (pipelineSteps.length <= 1) return;
        const newSteps = pipelineSteps.filter((_, i) => i !== index).map((s, i) => ({
            ...s,
            order: i + 1
        }));
        setPipelineSteps(newSteps);
    };

    const handleSubmit = async () => {
        if (!name) {
            setError('Curriculum name is required');
            return;
        }

        // Validate all steps have required configurations
        for (let i = 0; i < pipelineSteps.length; i++) {
            const step = pipelineSteps[i];
            if (!step.config.env_id) {
                setError(`Step ${i + 1}: Please select an environment`);
                return;
            }
            if (!step.config.reward_id) {
                setError(`Step ${i + 1}: Please select a reward function`);
                return;
            }
            if (step.step_type === 'training' || step.step_type === 'fine_tuning') {
                if (!step.config.agent_id) {
                    setError(`Step ${i + 1}: Please select an agent`);
                    return;
                }
            }
            if (step.step_type === 'residual') {
                if (!step.config.residual_connector_id) {
                    setError(`Step ${i + 1}: Please select a residual connector`);
                    return;
                }
            }
        }

        try {
            setLoading(true);
            const payload: any = {
                name,
                description,
                steps: pipelineSteps
            };
            // await api.createCurriculum(payload);
            setLoading(false);
            onSave();
            onClose();
        } catch (err) {
            setLoading(false);
            setError(err instanceof Error ? err.message : 'Failed to create curriculum');
        }
    };

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm">
           
        </div>
    );
}
*/
export default function CurriculumModal(props: any) { return null; }
