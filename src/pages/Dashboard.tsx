import { useEffect, useState } from 'react';
import { Activity, Beaker, BookOpen, Bot, CheckCircle2, Cog, FileCode2, Package, PauseCircle, Target, XCircle } from 'lucide-react';
import { Card, CardHeader } from '../components/common/Card';
import { api } from '../services/api';

interface DashboardStats {
  totals: {
    environments: number;
    agents: number;
    reward_functions: number;
    experiments: number;
    models: number;
    residual_connectors: number;
  };
  experiments_by_type: {
    [key: string]: number;
  };
  experiments_by_status: {
    [key: string]: number;
  };
}

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await api.get<DashboardStats>('/dashboard/stats');
        setStats(data);
      } catch (error) {
        console.error('Failed to fetch dashboard stats:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
    // Refresh every 5 seconds
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading || !stats) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="text-notion-text-secondary">Loading dashboard...</div>
      </div>
    );
  }

  const totalExperiments = stats.totals.experiments;
  const formatPercent = (value: number) => {
    if (totalExperiments === 0) return '0%';
    return `${Math.round((value / totalExperiments) * 100)}%`;
  };

  return (
    <div className="space-y-6">
      <div>
        <p className="text-xs uppercase tracking-wide text-notion-blue">Dashboard</p>
        <h1 className="text-2xl font-semibold text-notion-text">System Overview</h1>
        <p className="text-sm text-notion-text-secondary">
          Real-time statistics of models, experiments, and training progress across the system.
        </p>
      </div>

      {/* Entity Counts */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="space-y-3">
          <div className="flex items-center justify-between rounded-lg border border-notion-border bg-notion-light-gray p-4">
            <div>
              <p className="text-sm text-notion-text-secondary">Environments</p>
              <p className="text-3xl font-semibold text-notion-text">{stats.totals.environments}</p>
            </div>
            <div className="rounded-lg bg-blue-50 p-3 text-notion-blue">
              <Target className="h-6 w-6" />
            </div>
          </div>
        </Card>

        <Card className="space-y-3">
          <div className="flex items-center justify-between rounded-lg border border-notion-border bg-notion-light-gray p-4">
            <div>
              <p className="text-sm text-notion-text-secondary">Agents</p>
              <p className="text-3xl font-semibold text-notion-text">{stats.totals.agents}</p>
            </div>
            <div className="rounded-lg bg-purple-50 p-3 text-notion-purple">
              <Bot className="h-6 w-6" />
            </div>
          </div>
        </Card>

        <Card className="space-y-3">
          <div className="flex items-center justify-between rounded-lg border border-notion-border bg-notion-light-gray p-4">
            <div>
              <p className="text-sm text-notion-text-secondary">Reward Functions</p>
              <p className="text-3xl font-semibold text-notion-text">{stats.totals.reward_functions}</p>
            </div>
            <div className="rounded-lg bg-orange-50 p-3 text-notion-orange">
              <FileCode2 className="h-6 w-6" />
            </div>
          </div>
        </Card>

        <Card className="space-y-3">
          <div className="flex items-center justify-between rounded-lg border border-notion-border bg-notion-light-gray p-4">
            <div>
              <p className="text-sm text-notion-text-secondary">Model Snapshots</p>
              <p className="text-3xl font-semibold text-notion-text">{stats.totals.models}</p>
            </div>
            <div className="rounded-lg bg-green-50 p-3 text-notion-green">
              <Package className="h-6 w-6" />
            </div>
          </div>
        </Card>
      </div>

      {/* Experiments Overview */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card className="space-y-3">
          <CardHeader title="Total Experiments" />
          <div className="flex items-center justify-between rounded-lg border border-notion-border bg-notion-light-gray p-4">
            <div>
              <p className="text-sm text-notion-text-secondary">All experiment runs</p>
              <p className="text-3xl font-semibold text-notion-text">{stats.totals.experiments}</p>
              <p className="text-xs text-notion-text-tertiary">Training, simulation, and fine-tuning combined</p>
            </div>
            <div className="rounded-lg bg-yellow-50 p-3 text-notion-yellow">
              <Beaker className="h-8 w-8" />
            </div>
          </div>
        </Card>
      </div>

      {/* Experiments by Type */}
      <Card>
        <CardHeader title="Experiments by Type" />
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="rounded-lg border border-notion-border bg-notion-light-gray p-4">
            <div className="flex items-center justify-between text-xs text-notion-text-secondary mb-3">
              <span>Standard training runs</span>
              <span className="rounded-md px-2 py-0.5 text-[11px] bg-blue-50 text-notion-blue">
                {formatPercent(stats.experiments_by_type['Training'] || 0)}
              </span>
            </div>
            <div className="flex items-end justify-between">
              <div>
                <p className="text-sm text-notion-text-secondary">Training</p>
                <p className="text-3xl font-semibold text-notion-text">{stats.experiments_by_type['Training'] || 0}</p>
              </div>
              <Beaker className="h-8 w-8 text-notion-text-tertiary" />
            </div>
          </div>

          <div className="rounded-lg border border-notion-border bg-notion-light-gray p-4">
            <div className="flex items-center justify-between text-xs text-notion-text-secondary mb-3">
              <span>Test runs with trained models</span>
              <span className="rounded-md px-2 py-0.5 text-[11px] bg-green-50 text-notion-green">
                {formatPercent(stats.experiments_by_type['Simulation'] || 0)}
              </span>
            </div>
            <div className="flex items-end justify-between">
              <div>
                <p className="text-sm text-notion-text-secondary">Simulation</p>
                <p className="text-3xl font-semibold text-notion-text">{stats.experiments_by_type['Simulation'] || 0}</p>
              </div>
              <Activity className="h-8 w-8 text-notion-text-tertiary" />
            </div>
          </div>

          <div className="rounded-lg border border-notion-border bg-notion-light-gray p-4">
            <div className="flex items-center justify-between text-xs text-notion-text-secondary mb-3">
              <span>Refining pre-trained models</span>
              <span className="rounded-md px-2 py-0.5 text-[11px] bg-purple-50 text-notion-purple">
                {formatPercent(stats.experiments_by_type['Fine-Tuning'] || 0)}
              </span>
            </div>
            <div className="flex items-end justify-between">
              <div>
                <p className="text-sm text-notion-text-secondary">Fine-Tuning</p>
                <p className="text-3xl font-semibold text-notion-text">{stats.experiments_by_type['Fine-Tuning'] || 0}</p>
              </div>
              <Cog className="h-8 w-8 text-notion-text-tertiary" />
            </div>
          </div>
        </div>
      </Card>

      {/* Experiment Status */}
      <Card>
        <CardHeader title="Experiment Status" />
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-lg border border-green-200 bg-green-50 p-4 transition hover:border-green-300">
            <div className="flex items-center justify-between text-xs text-notion-text-secondary">
              <span>Successfully finished</span>
              <span className="rounded-md px-2 py-0.5 text-[11px] bg-green-100 text-notion-green">
                {formatPercent(stats.experiments_by_status['Completed'] || 0)}
              </span>
            </div>
            <div className="mt-3 flex items-end justify-between">
              <div>
                <p className="text-sm text-notion-green">Completed</p>
                <p className="text-3xl font-semibold text-notion-text">{stats.experiments_by_status['Completed'] || 0}</p>
              </div>
              <CheckCircle2 className="h-8 w-8 text-notion-green" />
            </div>
          </div>

          <div className="rounded-lg border border-orange-200 bg-orange-50 p-4 transition hover:border-orange-300">
            <div className="flex items-center justify-between text-xs text-notion-text-secondary">
              <span>Currently active</span>
              <span className="rounded-md px-2 py-0.5 text-[11px] bg-orange-100 text-notion-orange">
                {formatPercent(stats.experiments_by_status['In Progress'] || 0)}
              </span>
            </div>
            <div className="mt-3 flex items-end justify-between">
              <div>
                <p className="text-sm text-notion-orange">In Progress</p>
                <p className="text-3xl font-semibold text-notion-text">{stats.experiments_by_status['In Progress'] || 0}</p>
              </div>
              <Activity className="h-8 w-8 text-notion-orange" />
            </div>
          </div>

          <div className="rounded-lg border border-blue-200 bg-blue-50 p-4 transition hover:border-blue-300">
            <div className="flex items-center justify-between text-xs text-notion-text-secondary">
              <span>Temporarily paused</span>
              <span className="rounded-md px-2 py-0.5 text-[11px] bg-blue-100 text-notion-blue">
                {formatPercent(stats.experiments_by_status['Paused'] || 0)}
              </span>
            </div>
            <div className="mt-3 flex items-end justify-between">
              <div>
                <p className="text-sm text-notion-blue">Paused</p>
                <p className="text-3xl font-semibold text-notion-text">{stats.experiments_by_status['Paused'] || 0}</p>
              </div>
              <PauseCircle className="h-8 w-8 text-notion-blue" />
            </div>
          </div>

          <div className="rounded-lg border border-red-200 bg-red-50 p-4 transition hover:border-red-300">
            <div className="flex items-center justify-between text-xs text-notion-text-secondary">
              <span>Stopped or failed</span>
              <span className="rounded-md px-2 py-0.5 text-[11px] bg-red-100 text-notion-red">
                {formatPercent(stats.experiments_by_status['Cancelled'] || 0)}
              </span>
            </div>
            <div className="mt-3 flex items-end justify-between">
              <div>
                <p className="text-sm text-notion-red">Cancelled</p>
                <p className="text-3xl font-semibold text-notion-text">{stats.experiments_by_status['Cancelled'] || 0}</p>
              </div>
              <XCircle className="h-8 w-8 text-notion-red" />
            </div>
          </div>
        </div>
      </Card>

      {/* Additional System Info */}
      <Card>
        <CardHeader title="System Resources" />
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="rounded-lg border border-notion-border bg-notion-light-gray p-4">
            <p className="text-sm text-notion-text-secondary">Residual Connectors</p>
            <p className="text-2xl font-semibold text-notion-text mt-1">{stats.totals.residual_connectors}</p>
            <p className="text-xs text-notion-text-tertiary mt-1">Safety correction layers</p>
          </div>
          <div className="rounded-lg border border-notion-border bg-notion-light-gray p-4">
            <p className="text-sm text-notion-text-secondary">Active Components</p>
            <p className="text-2xl font-semibold text-notion-text mt-1">
              {stats.totals.environments + stats.totals.agents + stats.totals.reward_functions}
            </p>
            <p className="text-xs text-notion-text-tertiary mt-1">Environments, agents, and reward functions</p>
          </div>
        </div>
      </Card>
    </div>
  );
}
