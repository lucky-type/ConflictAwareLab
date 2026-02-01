import { Activity, Atom, Cpu, Orbit, Sparkles, Trophy } from 'lucide-react';

const toSeries = (values: number[]) => values.map((value, index) => ({ step: index * 10, value }));

export const metrics = {
  reward: toSeries([0, 12, 24, 37, 52, 61]),
};

export const statusCards = [
  {
    title: 'Active Environments',
    value: '12',
    change: '+2.3% vs last week',
    icon: Orbit,
    tone: 'cyan',
  },
  {
    title: 'Reward Trend',
    value: '↑ 61.4',
    change: 'Stable over 100 episodes',
    icon: Sparkles,
    tone: 'emerald',
  },
  {
    title: 'Running Experiments',
    value: '5',
    change: '3 scheduled today',
    icon: Activity,
    tone: 'amber',
  },
  {
    title: 'Model Registry',
    value: '38 models',
    change: '6 new this month',
    icon: Trophy,
    tone: 'violet',
  },
];

export const environments = [
  {
    name: 'Cityblock Flight v2',
    drones: '4 drones',
    status: 'Stable',
    reward: '+68.3',
    uptime: '99.2%',
    description: 'Tight turns and wind gust simulation.',
  },
  {
    name: 'Warehouse Picker',
    drones: '2 drones',
    status: 'Degraded',
    reward: '+54.1',
    uptime: '95.4%',
    description: 'High-density rack navigation tasks.',
  },
  {
    name: 'Maritime Dock Ops',
    drones: '3 drones',
    status: 'Stable',
    reward: '+71.8',
    uptime: '99.8%',
    description: 'Heavy wind shear and moving targets.',
  },
  {
    name: 'Canyon Run',
    drones: '1 drone',
    status: 'Initializing',
    reward: '+31.6',
    uptime: '—',
    description: 'Low-altitude evasive maneuvers.',
  },
];

export const algorithms = [
  {
    name: 'PPO-LSTM',
    version: 'v3.2.1',
    lastRun: '12 mins ago',
    status: 'Healthy',
    throughput: '2.8k FPS',
  },
  {
    name: 'SAC Hybrid',
    version: 'v2.9.0',
    lastRun: '1 hr ago',
    status: 'Warning',
    throughput: '1.2k FPS',
  },
  {
    name: 'DreamerV3',
    version: 'v1.7.4',
    lastRun: '4 hrs ago',
    status: 'Healthy',
    throughput: '980 FPS',
  },
  {
    name: 'A2C Lightweight',
    version: 'v5.1.0',
    lastRun: 'Yesterday',
    status: 'Deprecated',
    throughput: '450 FPS',
  },
];

export interface RewardFunctionDefinition {
  id: string;
  name: string;
  maintainer: string;
  status: string;
  lastUpdated: string;
  wProgress: number;
  wTime: number;
  wJerk: number;
  arrivalReward: number;
  crashPenalty: number;
}

export const rewardFunctions: RewardFunctionDefinition[] = [
  {
    id: 'rf-001',
    name: 'Collision Avoidance',
    maintainer: 'Dr. Rivera',
    status: 'Stable',
    lastUpdated: '2 days ago',
    wProgress: 2.6,
    wTime: 0.85,
    wJerk: 0.4,
    arrivalReward: 1200,
    crashPenalty: -900,
  },
  {
    id: 'rf-002',
    name: 'Energy Optimizer',
    maintainer: 'Kai',
    status: 'Experimental',
    lastUpdated: '1 hr ago',
    wProgress: 1.4,
    wTime: 0.35,
    wJerk: 0.3,
    arrivalReward: 900,
    crashPenalty: -400,
  },
  {
    id: 'rf-003',
    name: 'Smooth Landing',
    maintainer: 'Iris',
    status: 'Stable',
    lastUpdated: '4 days ago',
    wProgress: 2.1,
    wTime: 0.65,
    wJerk: 0.25,
    arrivalReward: 1000,
    crashPenalty: -700,
  },
  {
    id: 'rf-004',
    name: 'Adaptive Tracking',
    maintainer: 'Mina',
    status: 'Draft',
    lastUpdated: '8 hrs ago',
    wProgress: 3.1,
    wTime: 1.1,
    wJerk: 0.8,
    arrivalReward: 1300,
    crashPenalty: -1100,
  },
];

export const modelRegistry = [
  {
    experimentId: 'EXP-2412-A',
    experimentName: 'warehouse_navigation_rollout',
    owner: 'RL Ops',
    status: 'Ready',
    env: 'Warehouse Picker',
    algorithm: 'PPO-LSTM',
    startedAt: '12.12.2024 10:14 UTC',
    description: 'Серія тренувань на складі з контролем навантаження та мінімізацією зупинок.',
    snapshots: [
      {
        iteration: '60000',
        metric: '+61.8 reward',
        size: '205 MB',
        savedAt: '00:42:10 runtime',
        note: 'Перший стабільний чекпоінт для регресії.'
      },
      {
        iteration: '90000',
        metric: '+64.5 reward',
        size: '210 MB',
        savedAt: '01:18:20 runtime',
        note: 'Усунули коливання на різкому повороті.'
      },
      {
        iteration: '120000',
        metric: '+66.2 reward',
        size: '218 MB',
        savedAt: '01:44:10 runtime',
        note: 'Фінальний снапшот, готовий до розгортання.'
      },
    ],
  },
  {
    experimentId: 'EXP-2411-C',
    experimentName: 'maritime_gust_control',
    owner: 'Research',
    status: 'Beta',
    env: 'Maritime Dock Ops',
    algorithm: 'SAC Hybrid',
    startedAt: '08.12.2024 06:48 UTC',
    description: 'Модель для стійкого посадження на док при вітрі. З\'явиться після запуску повного сеансу тренувань.',
    snapshots: [
      {
        iteration: '45000',
        metric: '+58.1 reward',
        size: '330 MB',
        savedAt: '00:34:03 runtime',
        note: 'Захопили момент пориву вітру.'
      },
      {
        iteration: '78000',
        metric: '+60.4 reward',
        size: '335 MB',
        savedAt: '01:08:22 runtime',
        note: 'Стала кращою реакція на глісаду.'
      },
    ],
  },
  {
    experimentId: 'EXP-2410-E',
    experimentName: 'cityblock_autonomy_stack',
    owner: 'Applied ML',
    status: 'Ready',
    env: 'Cityblock Flight v2',
    algorithm: 'DreamerV3',
    startedAt: '01.12.2024 18:02 UTC',
    description: 'Повна автономія кварталу; моделі потрапляють сюди одразу після тренування.',
    snapshots: [
      {
        iteration: '30000',
        metric: '+52.4 reward',
        size: '275 MB',
        savedAt: '00:28:09 runtime',
        note: 'Чутливість до світлових умов.'
      },
      {
        iteration: '64000',
        metric: '+57.3 reward',
        size: '283 MB',
        savedAt: '00:59:15 runtime',
        note: 'Додали покращений планувальник траєкторій.'
      },
      {
        iteration: '101000',
        metric: '+62.7 reward',
        size: '295 MB',
        savedAt: '01:48:03 runtime',
        note: 'Кращий обхід перешкод у темряві.'
      },
    ],
  },
];

export const experiments = [
  {
    id: 'EXP-2412-A',
    title: 'Warehouse navigation rollout',
    type: 'Training',
    env: 'Warehouse Picker',
    algorithm: 'PPO-LSTM',
    rewardFunction: 'Collision Avoidance',
    status: 'Running',
    reward: '+54.1',
    duration: '01:18:42',
    owner: 'Maya',
    progress: 56,
    stepsPlanned: 1200000,
    snapshotInterval: 50000,
    activeStep: 680000,
    metrics: {
      avgReward: toSeries([12, 22, 34, 46, 58, 66, 74]),
      successRate: toSeries([42, 55, 61, 72, 81, 88, 91]),
      collisionRate: toSeries([18, 16, 13, 9, 6, 4, 3]),
      episodeLength: toSeries([34, 32, 31, 29, 28, 28, 27]),
      policyLoss: toSeries([1.2, 0.9, 0.7, 0.5, 0.36, 0.28, 0.21]),
      valueLoss: toSeries([1.8, 1.3, 0.92, 0.71, 0.5, 0.41, 0.38]),
      entropy: toSeries([0.92, 0.88, 0.81, 0.74, 0.69, 0.63, 0.6]),
    },
    logs: [
      { time: '12:04:11', text: '[Trainer] Bootstrapping PPO baseline' },
      { time: '12:04:23', text: '[Env] Warehouse Picker warmed up' },
      { time: '12:05:01', text: '[Rollout] Mean reward reached 45.1' },
      { time: '12:06:45', text: '[Checkpoint] Snapshot saved every 50k steps' },
      { time: '12:07:20', text: '[Policy] KL divergence within bounds' },
      { time: '12:08:02', text: '[Monitor] Success rate 88%' },
    ],
    timeline: [
      { label: 'Queued', time: '11:40' },
      { label: 'Launched', time: '12:04' },
      { label: 'Snapshot', time: '12:06' },
    ],
    lockedItems: [
      { label: 'Environment', value: 'Warehouse Picker' },
      { label: 'Algorithm', value: 'PPO-LSTM' },
      { label: 'Reward', value: 'Collision Avoidance' },
    ],
  },
  {
    id: 'EXP-2411-B',
    title: 'Dockside gust stress test',
    type: 'Simulation',
    env: 'Maritime Dock Ops',
    algorithm: 'SAC Hybrid',
    rewardFunction: 'Energy Optimizer',
    status: 'Running',
    reward: '+71.8',
    duration: '00:18:50',
    owner: 'Rui',
    progress: 62,
    episodes: 24,
    model: 'navmesh-v2.4',
    metrics: {
      avgReward: toSeries([68, 69, 70, 70, 71, 71.4, 72]),
      successRate: toSeries([78, 81, 84, 86, 88, 90, 92]),
      collisionRate: toSeries([2.4, 1.8, 1.1, 0.9, 0.7, 0.5, 0.4]),
      episodeLength: toSeries([44, 41, 38, 36, 34, 32, 32]),
      policyLoss: toSeries([0.7, 0.62, 0.55, 0.52, 0.48, 0.45, 0.44]),
      valueLoss: toSeries([1, 0.9, 0.82, 0.74, 0.68, 0.6, 0.55]),
      entropy: toSeries([0.77, 0.74, 0.7, 0.66, 0.63, 0.6, 0.58]),
    },
    simulationPath: [
      { x: 5, y: 80 },
      { x: 25, y: 68 },
      { x: 35, y: 50 },
      { x: 52, y: 38 },
      { x: 75, y: 32 },
      { x: 88, y: 12 },
    ],
    simulationObstacles: [
      { x: 32, y: 55, r: 6 },
      { x: 60, y: 30, r: 8 },
      { x: 80, y: 60, r: 5 },
    ],
    logs: [
      { time: '09:42:02', text: '[Simulation] Drone spawned at Dock NW entry' },
      { time: '09:42:11', text: '[Wind Model] Gusts level 0.6 applied' },
      { time: '09:42:33', text: '[Episode 3] Collision avoided with 1.2m margin' },
      { time: '09:43:10', text: '[Telemetry] Energy draw stabilized' },
      { time: '09:43:44', text: '[Episode 6] Destination reached in 34.2s' },
    ],
    timeline: [
      { label: 'Queued', time: '09:30' },
      { label: 'Replay', time: '09:38' },
      { label: 'Streaming', time: '09:42' },
    ],
    lockedItems: [
      { label: 'Environment', value: 'Maritime Dock Ops' },
      { label: 'Model', value: 'navmesh-v2.4 snapshot' },
    ],
  },
  {
    id: 'EXP-2410-D',
    title: 'Vision transformer fine-tune',
    type: 'Training',
    env: 'Cityblock Flight v2',
    algorithm: 'DreamerV3',
    rewardFunction: 'Adaptive Tracking',
    status: 'Completed',
    reward: '+66.3',
    duration: '03:44:10',
    owner: 'Amir',
    progress: 100,
    stepsPlanned: 900000,
    snapshotInterval: 30000,
    activeStep: 900000,
    metrics: {
      avgReward: toSeries([22, 32, 41, 58, 63, 65, 66]),
      successRate: toSeries([44, 57, 61, 74, 81, 88, 90]),
      collisionRate: toSeries([3.2, 2.8, 2.1, 1.4, 1, 0.7, 0.6]),
      episodeLength: toSeries([42, 39, 37, 35, 34, 33, 33]),
      policyLoss: toSeries([1.5, 1.2, 0.92, 0.61, 0.44, 0.32, 0.3]),
      valueLoss: toSeries([1.9, 1.4, 1.1, 0.8, 0.61, 0.52, 0.48]),
      entropy: toSeries([0.89, 0.84, 0.78, 0.72, 0.67, 0.63, 0.6]),
    },
    logs: [
      { time: '05:14:01', text: '[Trainer] Fine-tuning checkpoints ready' },
      { time: '05:44:19', text: '[Monitor] Avg reward 61.3 after 450k steps' },
      { time: '06:11:42', text: '[Diagnostics] Policy/value loss converged' },
      { time: '07:01:14', text: '[Artifacts] Snapshot exported to registry' },
    ],
    timeline: [
      { label: 'Queued', time: '04:50' },
      { label: 'Running', time: '05:10' },
      { label: 'Completed', time: '08:58' },
    ],
    lockedItems: [
      { label: 'Environment', value: 'Cityblock Flight v2' },
      { label: 'Algorithm', value: 'DreamerV3' },
      { label: 'Reward', value: 'Adaptive Tracking' },
    ],
  },
  {
    id: 'EXP-2409-F',
    title: 'Autonomy firmware soak',
    type: 'Training',
    env: 'Canyon Run',
    algorithm: 'PPO-LSTM',
    rewardFunction: 'Smooth Landing',
    status: 'Paused',
    reward: '+31.6',
    duration: '00:42:10',
    owner: 'Nora',
    progress: 48,
    stepsPlanned: 450000,
    snapshotInterval: 25000,
    activeStep: 220000,
    metrics: {
      avgReward: toSeries([12, 18, 27, 31, 32, 33, 33]),
      successRate: toSeries([32, 38, 45, 51, 56, 59, 60]),
      collisionRate: toSeries([4.2, 3.9, 3.2, 2.8, 2.3, 2.1, 2]),
      episodeLength: toSeries([52, 51, 48, 45, 44, 43, 43]),
      policyLoss: toSeries([1.8, 1.6, 1.2, 1.05, 0.9, 0.8, 0.8]),
      valueLoss: toSeries([2.1, 1.8, 1.5, 1.3, 1.19, 1.12, 1.1]),
      entropy: toSeries([0.95, 0.94, 0.9, 0.86, 0.83, 0.81, 0.81]),
    },
    logs: [
      { time: '18:01:11', text: '[Trainer] Canyon gust regime engaged' },
      { time: '18:12:44', text: '[Monitor] Divergence spikes from firmware patch' },
      { time: '18:14:08', text: '[Ops] Experiment paused for debug' },
    ],
    timeline: [
      { label: 'Queued', time: '17:40' },
      { label: 'Running', time: '18:00' },
      { label: 'Paused', time: '18:14' },
    ],
    lockedItems: [
      { label: 'Environment', value: 'Canyon Run' },
      { label: 'Algorithm', value: 'PPO-LSTM' },
      { label: 'Reward', value: 'Smooth Landing' },
    ],
  },
  {
    id: 'EXP-2409-G',
    title: 'Dock collision fail-safe',
    type: 'Simulation',
    env: 'Maritime Dock Ops',
    algorithm: 'SAC Hybrid',
    rewardFunction: 'Collision Avoidance',
    status: 'Canceled',
    reward: '—',
    duration: '—',
    owner: 'Ops',
    progress: 24,
    episodes: 12,
    model: 'precision-drop-v3',
    metrics: {
      avgReward: toSeries([42, 41, 39, 35, 32, 0, 0]),
      successRate: toSeries([60, 59, 57, 40, 0, 0, 0]),
      collisionRate: toSeries([3.4, 3.6, 3.8, 5.1, 7.4, 8, 8]),
      episodeLength: toSeries([40, 38, 34, 28, 23, 18, 18]),
      policyLoss: toSeries([0.9, 0.95, 1, 1.2, 1.4, 1.4, 1.4]),
      valueLoss: toSeries([1.2, 1.3, 1.45, 1.62, 1.8, 1.8, 1.8]),
      entropy: toSeries([0.8, 0.79, 0.76, 0.65, 0.5, 0.5, 0.5]),
    },
    simulationPath: [
      { x: 10, y: 85 },
      { x: 30, y: 70 },
      { x: 40, y: 58 },
      { x: 62, y: 52 },
      { x: 78, y: 40 },
    ],
    simulationObstacles: [
      { x: 45, y: 60, r: 8 },
      { x: 70, y: 45, r: 6 },
    ],
    logs: [
      { time: '10:21:11', text: '[Simulation] Snapshot precision-drop-v3 loaded' },
      { time: '10:22:01', text: '[Episode 2] Drone clipped dock crane (minor)' },
      { time: '10:22:44', text: '[Safety] Abort requested by Ops' },
    ],
    timeline: [
      { label: 'Queued', time: '10:05' },
      { label: 'Running', time: '10:20' },
      { label: 'Canceled', time: '10:23' },
    ],
    lockedItems: [
      { label: 'Environment', value: 'Maritime Dock Ops' },
      { label: 'Model', value: 'precision-drop-v3 snapshot' },
    ],
  },
];

export const comparison = [
  { metric: 'Success Rate', baseline: 78, candidate: 84 },
  { metric: 'Mean Reward', baseline: 61, candidate: 68 },
  { metric: 'Collisions / hr', baseline: 1.8, candidate: 0.9 },
  { metric: 'Energy Draw', baseline: 92, candidate: 81 },
];

export const statuses = {
  Stable: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30',
  Healthy: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30',
  Degraded: 'text-amber-400 bg-amber-400/10 border-amber-400/30',
  Experimental: 'text-sky-300 bg-sky-300/10 border-sky-300/30',
  Warning: 'text-amber-400 bg-amber-400/10 border-amber-400/30',
  Initializing: 'text-cyan-300 bg-cyan-300/10 border-cyan-300/30',
  Beta: 'text-purple-300 bg-purple-300/10 border-purple-300/30',
  Ready: 'text-emerald-300 bg-emerald-300/10 border-emerald-300/30',
  Draft: 'text-slate-300 bg-slate-300/10 border-slate-300/30',
  Queued: 'text-slate-200 bg-slate-200/10 border-slate-200/30',
  Completed: 'text-indigo-200 bg-indigo-200/10 border-indigo-200/30',
  Running: 'text-amber-300 bg-amber-300/10 border-amber-300/30',
  Paused: 'text-sky-300 bg-sky-300/10 border-sky-300/30',
  Canceled: 'text-rose-300 bg-rose-300/10 border-rose-300/30',
};

export const quickActions = [
  { label: 'Calibrate sensors', icon: Atom },
  { label: 'Run diagnostics', icon: Cpu },
  { label: 'Sync fleet', icon: Sparkles },
];
