import { Activity, Atom, Cpu, Orbit, Sparkles, Trophy } from 'lucide-react';

export const metrics = {
  reward: [
    { step: 0, value: 0 },
    { step: 20, value: 12 },
    { step: 40, value: 24 },
    { step: 60, value: 37 },
    { step: 80, value: 52 },
    { step: 100, value: 61 },
  ],
  stability: [
    { step: 0, value: 0 },
    { step: 20, value: 30 },
    { step: 40, value: 40 },
    { step: 60, value: 63 },
    { step: 80, value: 71 },
    { step: 100, value: 82 },
  ],
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

export const rewardFunctions = [
  {
    name: 'Collision Avoidance',
    author: 'Dr. Rivera',
    status: 'Stable',
    lastUpdated: '2 days ago',
  },
  {
    name: 'Energy Optimizer',
    author: 'Kai',
    status: 'Experimental',
    lastUpdated: '1 hr ago',
  },
  {
    name: 'Smooth Landing',
    author: 'Iris',
    status: 'Stable',
    lastUpdated: '4 days ago',
  },
  {
    name: 'Adaptive Tracking',
    author: 'Mina',
    status: 'Draft',
    lastUpdated: '8 hrs ago',
  },
];

export const models = [
  {
    name: 'navmesh-v2.4',
    owner: 'RL Ops',
    size: '218 MB',
    status: 'Ready',
    tags: ['vision', 'navigation'],
    metric: '+12.8 reward',
  },
  {
    name: 'aerial-foundation-v1',
    owner: 'Research',
    size: '512 MB',
    status: 'Beta',
    tags: ['foundation', 'multimodal'],
    metric: '+19.6 reward',
  },
  {
    name: 'precision-drop-v3',
    owner: 'Applied ML',
    size: '341 MB',
    status: 'Ready',
    tags: ['control', 'delivery'],
    metric: '+9.1 reward',
  },
];

export const experiments = [
  {
    id: 'EXP-2412-A',
    title: 'Warehouse navigation rollout',
    env: 'Warehouse Picker',
    algorithm: 'PPO-LSTM',
    status: 'Running',
    reward: '+54.1',
    duration: '01:18:42',
    owner: 'Maya',
  },
  {
    id: 'EXP-2411-B',
    title: 'Dockside gust stress test',
    env: 'Maritime Dock Ops',
    algorithm: 'SAC Hybrid',
    status: 'Queued',
    reward: '+71.8',
    duration: '—',
    owner: 'Rui',
  },
  {
    id: 'EXP-2410-D',
    title: 'Vision transformer fine-tune',
    env: 'Cityblock Flight v2',
    algorithm: 'DreamerV3',
    status: 'Completed',
    reward: '+66.3',
    duration: '03:44:10',
    owner: 'Amir',
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
};

export const quickActions = [
  { label: 'Calibrate sensors', icon: Atom },
  { label: 'Run diagnostics', icon: Cpu },
  { label: 'Sync fleet', icon: Sparkles },
];
