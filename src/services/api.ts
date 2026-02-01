/**
 * API client for ConflictAwareLab backend
 */

const API_BASE_URL = 'http://localhost:8000';

// ==================== Types ====================
export interface ObstacleConfig {
  type: string;
  diameter: number;
  speed: number;
  strategy: string;
  chaos: number;
}

export interface Environment {
  id: number;
  name: string;
  length: number;
  width: number;
  height: number;
  buffer_start: number;
  buffer_end: number;
  target_diameter: number;
  obstacles: ObstacleConfig[];
  is_locked: boolean;
  is_predicted: boolean;
  created_at: string;
}

export interface EnvironmentCreate {
  name: string;
  length: number;
  width: number;
  height: number;
  buffer_start: number;
  buffer_end: number;
  target_diameter: number;
  obstacles: ObstacleConfig[];
  is_predicted?: boolean;
}

export interface Agent {
  id: number;
  name: string;
  type: string;
  agent_diameter: number;
  agent_max_speed: number;
  lidar_range: number;
  lidar_rays: number;
  kinematic_type: string;
  parameters: Record<string, any>;
  is_locked: boolean;
  created_at: string;
}

export interface AgentCreate {
  name: string;
  type: string;
  agent_diameter: number;
  agent_max_speed: number;
  lidar_range: number;
  lidar_rays: number;
  kinematic_type: string;
  parameters: Record<string, any>;
}

export interface ResidualConnector {
  id: number;
  name: string;
  parent_agent_id: number;
  algorithm: string;
  net_arch: number[];
  parameters: Record<string, any>;
  k_factor: number;
  adaptive_k: boolean;  // Enable adaptive K-factor (CARS)
  enable_k_conf: boolean;  // CARS: Enable conflict-aware component
  enable_k_risk: boolean;  // CARS: Enable risk-aware component
  // SHIELDING COMMENTED OUT
  // enable_k_shield: boolean;  // CARS: Enable shield-aware component
  is_locked: boolean;
  created_at: string;
  parent_agent?: Agent;
}

export interface ResidualConnectorCreate {
  name: string;
  parent_agent_id: number;
  algorithm: string;
  net_arch: number[];
  parameters: Record<string, any>;
  k_factor: number;
  adaptive_k?: boolean;  // Enable adaptive K-factor (CARS) (default: false)
  enable_k_conf?: boolean;  // CARS: Enable conflict-aware component (default: true)
  enable_k_risk?: boolean;  // CARS: Enable risk-aware component (default: true)
  // SHIELDING COMMENTED OUT
  // enable_k_shield?: boolean;  // CARS: Enable shield-aware component (default: true)
}

export interface RewardFunction {
  id: number;
  name: string;
  w_progress: number;
  w_time: number;
  w_jerk: number;
  success_reward: number;
  crash_reward: number;
  is_locked: boolean;
  created_at: string;
}

export interface RewardFunctionCreate {
  name: string;
  w_progress: number;
  w_time: number;
  w_jerk: number;
  success_reward: number;
  crash_reward: number;
}

export type ExperimentType = 'Training' | 'Simulation' | 'Fine-Tuning' | 'Evaluation';
export type ExperimentStatus = 'In Progress' | 'Completed' | 'Paused' | 'Cancelled';

export interface SafetyConstraintConfig {
  enabled: boolean;
  risk_budget: number;
  initial_lambda: number;
  lambda_learning_rate: number;
  update_frequency: number;
  // Lambda warmup schedule
  target_lambda?: number | null;
  warmup_episodes?: number;
  warmup_schedule?: string;
  // Cost signal configuration
  cost_signal_preset: string;
  collision_weight: number;
  near_miss_weight: number;
  danger_zone_weight: number;
  near_miss_threshold: number;
  ignore_walls: boolean;
  wall_near_miss_weight: number;
  wall_danger_zone_weight: number;
  wall_near_miss_threshold: number;
  // SHIELDING COMMENTED OUT
  // Safety Shield (runtime action filtering)
  // shield_enabled?: boolean;
  // shield_threshold?: number;
  // shield_fallback_strength?: number;
}

export interface Experiment {
  id: number;
  name: string;
  type: ExperimentType;
  env_id: number;
  agent_id: number;
  reward_id: number;
  status: ExperimentStatus;
  total_steps: number;
  current_step: number;
  snapshot_freq: number;
  model_snapshot_id?: number;
  base_model_snapshot_id?: number;
  residual_connector_id?: number;
  safety_constraint?: SafetyConstraintConfig;
  trajectory_data?: Array<Record<string, any>>;
  evaluation_episodes?: number;
  fps_delay?: number;
  seed?: number | null;  // Random seed for reproducibility
  created_at: string;
  environment?: Environment;
  agent?: Agent;
  reward_function?: RewardFunction;
}

export interface ExperimentCreate {
  name: string;
  type: ExperimentType;
  env_id: number;
  agent_id: number;
  reward_id: number;
  total_steps: number;
  snapshot_freq: number;
  max_ep_length: number;
  model_snapshot_id?: number;
  base_model_snapshot_id?: number;
  fine_tuning_strategy?: string;
  training_mode?: 'standard' | 'residual';
  residual_base_model_id?: number;
  residual_connector_id?: number;
  safety_constraint?: SafetyConstraintConfig;
  evaluation_episodes?: number;
  fps_delay?: number;
  seed?: number | null;  // Random seed for reproducibility
}

export interface ModelSnapshot {
  id: number;
  experiment_id: number;
  iteration: number;
  file_path: string;
  metrics_at_save: Record<string, any>;
  created_at: string;
}

export interface ExperimentMetric {
  id: number;
  experiment_id: number;
  step: number;
  values: Record<string, any>;
  logs?: string;
  created_at: string;
}

// ==================== Real-time Updates ====================
export interface TrainingMetrics {
  step: number;
  episode_count: number;

  // A) Performance Metrics
  reward_raw_mean: number;
  reward_raw_std: number;
  reward_shaped_mean: number;
  success_rate: number;  // Percentage
  crash_rate: number;  // Percentage
  mean_ep_length: number;
  ep_length_std: number;

  // B) Safety/Risk Metrics
  cost_mean: number;
  cost_std: number;
  near_miss_mean: number;
  danger_time_mean: number;
  violation_rate: number;  // Percentage
  // SHIELDING COMMENTED OUT
  // shield_intervention_rate?: number;  // Percentage of steps with shield intervention
  // shield_interventions_per_episode?: number;  // Avg shield interventions per episode

  // C) Lagrangian Dynamics
  lambda: number;
  epsilon: number;
  warmup_target_lambda?: number | null;
  warmup_progress?: number | null;

  // D) Legacy/compatibility
  mean_reward?: number;

  // E) Algorithm-specific (optional)
  policy_loss?: number;
  value_loss?: number;
  actor_loss?: number;
  critic_loss?: number;
  entropy?: number;
  learning_rate?: number;

  // F) Residual Learning
  residual_correction_magnitude?: number;
  residual_base_magnitude?: number;
  residual_intervention_ratio?: number;

  // G) Additional metrics
  timeout_rate?: number;

  // H) Evaluation Metrics (New)
  total_episodes?: number;
  total_successes?: number;
  total_failures?: number;
  total_timeouts?: number;

  // I) Advanced Residual Metrics
  residual_contribution_mean?: number;
  conflict_mean?: number;
  conflict_std?: number;
  intervention_rate?: number;
  effective_k_mean?: number;  // Adaptive K-factor mean value

  // J) CARS K Components
  k_conf_mean?: number;   // Conflict-aware K component
  k_risk_mean?: number;   // Risk (\u03bb-based) K component
  // SHIELDING COMMENTED OUT
  // k_shield_mean?: number; // Shield intervention K component
}

export interface ExperimentProgress {
  experiment_id: number;
  step: number;
  total_steps: number;
  episode?: number; // For evaluation mode
  total_episodes?: number; // For evaluation mode
  metrics: Partial<TrainingMetrics>;
  status: ExperimentStatus;
  timestamp: number;
}

export interface SimulationFrame {
  experiment_id: number;
  frame: number;
  timestamp: number;
  agent_position: { x: number; y: number; z: number };
  agent_velocity?: { x: number; y: number; z: number };
  agent_orientation?: { roll: number; pitch: number; yaw: number };
  target_position: { x: number; y: number; z: number };
  obstacles: Array<{
    x: number;
    y: number;
    z: number;
    diameter: number;
    speed: number;
    strategy?: string;
    chaos?: number;
    vx?: number;
    vy?: number;
    vz?: number;
    [key: string]: any;
  }>;
  lidar_readings: number[];
  lidar_hit_info?: Array<{
    distance: number;
    type: string;
    hit_position: [number, number, number] | null;
    direction?: [number, number, number];
  }>;
  reward: number;
  done: boolean;
  success: boolean;
  crashed: boolean;
  crash_type?: string;
  episode?: number;
}

export interface JobCapacity {
  running_jobs: number;
  max_concurrent_jobs: number;
  available_slots: number;
  can_start_new: boolean;
}

// // ==================== Curriculum Learning ====================
// export type CurriculumStatus = 'Pending' | 'Running' | 'Paused' | 'Completed' | 'Failed';
// export type CurriculumStepType = 'training' | 'fine_tuning' | 'residual';

// export interface CurriculumStepConfig {
//   env_id: number;
//   agent_id?: number;
//   residual_connector_id?: number;
//   reward_id: number;
//   total_steps: number;
//   snapshot_freq: number;
//   max_ep_length: number;
// }

// export interface CurriculumStep {
//   id: number;
//   curriculum_id: number;
//   order: number;
//   step_type: CurriculumStepType;
//   config: CurriculumStepConfig;
//   safety_config?: SafetyConstraintConfig;
//   target_success_rate: number;
//   min_episodes: number;
//   is_completed: boolean;
// }

// export interface CurriculumStepCreate {
//   order: number;
//   step_type: CurriculumStepType;
//   config: CurriculumStepConfig;
//   safety_config?: SafetyConstraintConfig;
//   target_success_rate: number;
//   min_episodes: number;
// }

// export interface Curriculum {
//   id: number;
//   name: string;
//   description?: string;
//   status: CurriculumStatus;
//   current_step_order: number;
//   created_at: string;
//   updated_at?: string;
//   steps: CurriculumStep[];
// }

// export interface CurriculumCreate {
//   name: string;
//   description?: string;
//   steps: CurriculumStepCreate[];
// }

// ==================== API Client ====================
class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `API Error: ${response.status}`);
    }

    return response.json();
  }

  // Generic GET method
  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint);
  }

  // Generic POST method
  async post<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  // Generic PUT method
  async put<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  // Generic DELETE method
  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'DELETE',
    });
  }

  // ==================== Environments ====================
  async getEnvironments(): Promise<Environment[]> {
    return this.request('/environments');
  }

  async getEnvironment(id: number): Promise<Environment> {
    return this.request(`/environments/${id}`);
  }

  async createEnvironment(data: EnvironmentCreate): Promise<Environment> {
    return this.request('/environments', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateEnvironment(id: number, data: Partial<EnvironmentCreate>): Promise<Environment> {
    return this.request(`/environments/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteEnvironment(id: number): Promise<{ status: string }> {
    return this.request(`/environments/${id}`, {
      method: 'DELETE',
    });
  }

  // ==================== Agents ====================
  async getAgents(): Promise<Agent[]> {
    return this.request('/agents');
  }

  async getAgent(id: number): Promise<Agent> {
    return this.request(`/agents/${id}`);
  }

  async createAgent(data: AgentCreate): Promise<Agent> {
    return this.request('/agents', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateAgent(id: number, data: Partial<AgentCreate>): Promise<Agent> {
    return this.request(`/agents/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteAgent(id: number): Promise<{ status: string }> {
    return this.request(`/agents/${id}`, {
      method: 'DELETE',
    });
  }

  // ==================== Residual Connectors ====================
  async getResidualConnectors(): Promise<ResidualConnector[]> {
    return this.request('/residual-connectors');
  }

  async getResidualConnector(id: number): Promise<ResidualConnector> {
    return this.request(`/residual-connectors/${id}`);
  }

  async createResidualConnector(data: ResidualConnectorCreate): Promise<ResidualConnector> {
    return this.request('/residual-connectors', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateResidualConnector(id: number, data: Partial<ResidualConnectorCreate>): Promise<ResidualConnector> {
    return this.request(`/residual-connectors/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteResidualConnector(id: number): Promise<{ status: string }> {
    return this.request(`/residual-connectors/${id}`, {
      method: 'DELETE',
    });
  }

  // ==================== Reward Functions ====================
  async getRewardFunctions(): Promise<RewardFunction[]> {
    return this.request('/reward-functions');
  }

  async getRewardFunction(id: number): Promise<RewardFunction> {
    return this.request(`/reward-functions/${id}`);
  }

  async createRewardFunction(data: RewardFunctionCreate): Promise<RewardFunction> {
    return this.request('/reward-functions', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateRewardFunction(id: number, data: Partial<RewardFunctionCreate>): Promise<RewardFunction> {
    return this.request(`/reward-functions/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteRewardFunction(id: number): Promise<{ status: string }> {
    return this.request(`/reward-functions/${id}`, {
      method: 'DELETE',
    });
  }

  // ==================== Experiments ====================
  async getExperiments(): Promise<Experiment[]> {
    return this.request('/experiments');
  }

  async getExperiment(id: number): Promise<Experiment> {
    return this.request(`/experiments/${id}`);
  }

  async createExperiment(data: ExperimentCreate): Promise<Experiment> {
    return this.request('/experiments', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateExperiment(id: number, data: Partial<ExperimentCreate>): Promise<Experiment> {
    return this.request(`/experiments/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteExperiment(id: number): Promise<{ status: string }> {
    return this.request(`/experiments/${id}`, {
      method: 'DELETE',
    });
  }

  // ==================== Model Snapshots ====================
  async getAllSnapshots(): Promise<ModelSnapshot[]> {
    return this.request('/snapshots');
  }

  async getExperimentSnapshots(experimentId: number): Promise<ModelSnapshot[]> {
    return this.request(`/experiments/${experimentId}/snapshots`);
  }

  async deleteSnapshot(snapshotId: number): Promise<{ message: string; snapshot_id: number }> {
    return this.request(`/snapshots/${snapshotId}`, {
      method: 'DELETE',
    });
  }

  async deleteSnapshots(snapshotIds: number[]): Promise<{
    message: string;
    deleted_ids: number[];
    failed_ids: Array<{ id: number; error: string }>;
  }> {
    return this.request('/snapshots/bulk-delete', {
      method: 'POST',
      body: JSON.stringify(snapshotIds),
    });
  }

  // ==================== Experiment Metrics ====================
  async getExperimentMetrics(experimentId: number, limit?: number): Promise<ExperimentMetric[]> {
    const params = limit ? `?limit=${limit}` : '';
    return this.request(`/experiments/${experimentId}/metrics${params}`);
  }

  async getBulkLatestMetrics(experimentIds: number[]): Promise<Record<number, ExperimentMetric>> {
    if (experimentIds.length === 0) return {};
    return this.request('/experiments/bulk-latest-metrics', {
      method: 'POST',
      body: JSON.stringify(experimentIds),
    });
  }

  // ==================== Experiment Control ====================
  async startExperiment(id: number): Promise<{ status: string; message: string }> {
    return this.request(`/experiments/${id}/start`, {
      method: 'POST',
    });
  }

  async pauseExperiment(id: number): Promise<{ status: string; message: string }> {
    return this.request(`/experiments/${id}/pause`, {
      method: 'POST',
    });
  }

  async resumeExperiment(id: number): Promise<{ status: string; message: string }> {
    return this.request(`/experiments/${id}/resume`, {
      method: 'POST',
    });
  }

  async cancelExperiment(id: number): Promise<{ status: string; message: string }> {
    return this.request(`/experiments/${id}/cancel`, {
      method: 'POST',
    });
  }

  async regenerateSimulation(id: number): Promise<{ status: string; message: string }> {
    return this.request(`/experiments/${id}/regenerate`, {
      method: 'POST',
    });
  }

  // ==================== Job Capacity ====================
  async getJobCapacity(): Promise<JobCapacity> {
    return this.request('/job-capacity');
  }

  // // ==================== Curriculums ====================
  // async getCurriculums(): Promise<Curriculum[]> {
  //   return this.request('/curriculums');
  // }

  // async getCurriculum(id: number): Promise<Curriculum> {
  //   return this.request(`/curriculums/${id}`);
  // }

  // async createCurriculum(data: CurriculumCreate): Promise<Curriculum> {
  //   return this.request('/curriculums', {
  //     method: 'POST',
  //     body: JSON.stringify(data),
  //   });
  // }

  // async startCurriculum(id: number): Promise<{ status: string; message: string }> {
  //   return this.request(`/curriculums/${id}/start`, {
  //     method: 'POST',
  //   });
  // }

  // async pauseCurriculum(id: number): Promise<{ status: string; message: string }> {
  //   return this.request(`/curriculums/${id}/pause`, {
  //     method: 'POST',
  //   });
  // }

  // async deleteCurriculum(id: number): Promise<{ status: string }> {
  //   return this.request(`/curriculums/${id}`, {
  //     method: 'DELETE',
  //   });
  // }

  // ==================== Trajectory & Replay ====================
  async getExperimentTrajectory(experimentId: number): Promise<{
    experiment_id: number;
    trajectory: Array<Record<string, any>>;
    total_frames: number;
  }> {
    return this.request(`/experiments/${experimentId}/trajectory`);
  }
}

export const api = new ApiClient();

// ==================== WebSocket Client ====================
export class ExperimentWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: number | null = null;
  private messageHandlers: Map<string, Set<(data: any) => void>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;
  private experimentId: number | null = null;

  connect(experimentId: number) {
    this.experimentId = experimentId;
    const wsUrl = `ws://localhost:8000/experiments/${experimentId}/ws`;

    // Only log on first attempt to avoid console spam
    if (this.reconnectAttempts === 0) {
      console.log(`Attempting WebSocket connection for experiment ${experimentId}`);
    }

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log(`WebSocket connected for experiment ${experimentId}`);
      this.reconnectAttempts = 0; // Reset on successful connection
    };

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        console.log('[WebSocket Message]', message);
        const handlers = this.messageHandlers.get(message.type);
        if (handlers) {
          handlers.forEach(handler => handler(message.data));
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    this.ws.onerror = () => {
      // Only log error on first attempt to avoid spam
      if (this.reconnectAttempts === 0) {
        console.warn('WebSocket connection failed (backend may not be running)');
      }
    };

    this.ws.onclose = () => {
      this.reconnectAttempts++;

      // Only auto-reconnect if we haven't exceeded max attempts
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectTimer = window.setTimeout(() => {
          if (this.experimentId !== null) {
            this.connect(this.experimentId);
          }
        }, 3000);
      } else if (this.reconnectAttempts === this.maxReconnectAttempts) {
        console.info('WebSocket backend unavailable. Real-time updates disabled.');
      }
    };
  }

  on(eventType: string, handler: (data: any) => void) {
    if (!this.messageHandlers.has(eventType)) {
      this.messageHandlers.set(eventType, new Set());
    }
    this.messageHandlers.get(eventType)!.add(handler);
  }

  off(eventType: string, handler: (data: any) => void) {
    const handlers = this.messageHandlers.get(eventType);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.messageHandlers.clear();
    this.reconnectAttempts = 0;
    this.experimentId = null;
  }
}
