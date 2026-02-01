"""Custom Drone Environment with Swarm Obstacles."""
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import time


class SwarmDrone:
    """Individual swarm drone with autonomous movement."""
    
    def __init__(self, position, corridor_length, corridor_width, corridor_height, config=None, physics_client=None):
        self.corridor_length = corridor_length
        self.corridor_width = corridor_width
        self.corridor_height = corridor_height
        self.physics_client = physics_client  # Store physics client ID
        
        # Use config if provided, otherwise use defaults
        if config:
            self.diameter = config.get('diameter', 0.5)
            self.max_speed = config.get('speed', 3.0)
            self.max_acceleration = config.get('acceleration', 1.5)
            self.chaos = config.get('chaos', 0.15)
            self.strategy = config.get('strategy', 'random')
        else:
            # Individual characteristics (random if no config)
            self.diameter = 0.5 + np.random.random() * 0.3
            self.max_speed = 2.0 + np.random.random() * 3.0
            self.max_acceleration = 1.0 + np.random.random() * 1.0
            self.chaos = 0.15
            self.strategy = 'random'
        
        self.personal_space_radius = self.diameter * 2.5
        
        # Strategy-specific state
        self.time = 0.0
        self.initial_position = np.array(position)
        self.pattern_center = np.array(position)
        self.pattern_radius = 5.0 + np.random.random() * 5.0  # For circular patterns
        self.pattern_phase = np.random.random() * 2 * np.pi  # Random starting phase
        self.zigzag_amplitude = 3.0 + np.random.random() * 3.0
        self.linear_direction = None  # Will be set for linear strategy
        
        # Create visual shape
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.diameter / 2,
            rgbaColor=[1, 0, 0, 1],
            physicsClientId=self.physics_client
        )
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.diameter / 2,
            physicsClientId=self.physics_client
        )
        
        self.id = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.physics_client
        )
        
        # Initialize with random velocity
        initial_velocity = (np.random.random(3) - 0.5) * self.max_speed
        p.resetBaseVelocity(self.id, linearVelocity=initial_velocity, physicsClientId=self.physics_client)
        
        # Random goal
        self.goal = self.generate_random_goal()
        
        # Chaos State (Correlated Random Walk)
        # Initialize random velocity component for smooth chaotic movement
        rand_vec = (np.random.random(3) - 0.5) * 2
        self.random_velocity_component = rand_vec / (np.linalg.norm(rand_vec) + 1e-6) * self.max_speed
        self.random_update_timer = 0.0
        self.random_update_interval = 0.5  # Update random vector every 0.5s
    
    def generate_random_goal(self):
        """Generate random goal within corridor bounds."""
        margin = 1.0
        return np.array([
            -self.corridor_width/2 + margin + np.random.random() * (self.corridor_width - 2 * margin),
            margin + np.random.random() * (self.corridor_height - 2 * margin),
            margin + np.random.random() * (self.corridor_length - 2 * margin)
        ])
    
    def update(self, all_drones, dt):
        """Update drone position using strategy-based movement with chaos."""
        pos, _ = p.getBasePositionAndOrientation(self.id, physicsClientId=self.physics_client)
        position = np.array(pos)
        vel, _ = p.getBaseVelocity(self.id, physicsClientId=self.physics_client)
        velocity = np.array(vel)
        
        self.time += dt
        
        # Get strategic desired velocity based on movement strategy
        if self.strategy == 'linear':
            strategic_velocity = self._get_linear_velocity(position)
        elif self.strategy == 'circular':
            strategic_velocity = self._get_circular_velocity(position)
        elif self.strategy == 'zigzag':
            strategic_velocity = self._get_zigzag_velocity(position)
        else:  # 'random' or default
            strategic_velocity = self._get_random_velocity(position)
        
        # Step 2: Separation - avoid other drones (applies to all strategies)
        separation_force = np.zeros(3)
        for other in all_drones:
            if other.id == self.id:
                continue
            
            other_pos, _ = p.getBasePositionAndOrientation(other.id, physicsClientId=self.physics_client)
            other_position = np.array(other_pos)
            distance = np.linalg.norm(position - other_position)
            
            if 0 < distance < self.personal_space_radius:
                repulsion = (position - other_position) / (distance + 1e-6)
                repulsion *= (self.personal_space_radius - distance) / self.personal_space_radius
                separation_force += repulsion
        
        # Step 3: Wall avoidance (applies to all strategies)
        wall_force = np.zeros(3)
        buffer_zone = 2.0
        
        dist_left = position[0] + self.corridor_width / 2
        dist_right = self.corridor_width / 2 - position[0]
        dist_bottom = position[1]
        dist_top = self.corridor_height - position[1]
        dist_back = position[2]
        dist_front = self.corridor_length - position[2]
        
        if dist_left < buffer_zone:
            wall_force[0] += (buffer_zone - dist_left) / buffer_zone
        if dist_right < buffer_zone:
            wall_force[0] -= (buffer_zone - dist_right) / buffer_zone
        if dist_bottom < buffer_zone:
            wall_force[1] += (buffer_zone - dist_bottom) / buffer_zone
        if dist_top < buffer_zone:
            wall_force[1] -= (buffer_zone - dist_top) / buffer_zone
        if dist_back < buffer_zone:
            wall_force[2] += (buffer_zone - dist_back) / buffer_zone
        if dist_front < buffer_zone:
            wall_force[2] -= (buffer_zone - dist_front) / buffer_zone
        
        # Step 4: Blend Strategy with Chaos (Correlated Random Walk)
        # Update random component periodically to create "wandering" behavior
        self.random_update_timer += dt
        if self.random_update_timer >= self.random_update_interval:
            self.random_update_timer = 0
            # Pick new random direction
            rand_vec = (np.random.random(3) - 0.5) * 2
            self.random_velocity_component = rand_vec / (np.linalg.norm(rand_vec) + 1e-6) * self.max_speed

        # Blend base strategy with random component
        # chaos=0 -> 100% strategy, chaos=1 -> 100% random drift
        blended_strategic_velocity = (1.0 - self.chaos) * strategic_velocity + self.chaos * self.random_velocity_component

        w_wall = 1.0
        w_separation = 0.7
        w_strategy = 0.5
        
        # Use blended strategy in the smart vector (preserves wall/separation avoidance)
        smart_vector = (wall_force * w_wall + 
                       separation_force * w_separation + 
                       blended_strategic_velocity * w_strategy)
        
        final_desired_velocity = smart_vector
        
        # Step 5: Apply physics
        acceleration = (final_desired_velocity - velocity) * self.max_acceleration
        new_velocity = velocity + acceleration * dt
        
        # Clamp to max speed
        speed = np.linalg.norm(new_velocity)
        if speed > self.max_speed:
            new_velocity = new_velocity / speed * self.max_speed
        
        p.resetBaseVelocity(self.id, linearVelocity=new_velocity.tolist(), physicsClientId=self.physics_client)
        
        # Hard boundary enforcement
        if (position[0] < -self.corridor_width/2 + 0.5 or 
            position[0] > self.corridor_width/2 - 0.5 or
            position[1] < 0.5 or position[1] > self.corridor_height - 0.5 or
            position[2] < 0.5 or position[2] > self.corridor_length - 0.5):
            # Reset to safe position if out of bounds
            safe_pos = np.clip(position, 
                             [-self.corridor_width/2 + 0.5, 0.5, 0.5],
                             [self.corridor_width/2 - 0.5, self.corridor_height - 0.5, self.corridor_length - 0.5])
            p.resetBasePositionAndOrientation(self.id, safe_pos.tolist(), [0, 0, 0, 1], physicsClientId=self.physics_client)
    
    def _get_linear_velocity(self, position):
        """Linear movement: move in a straight line along the corridor."""
        if self.linear_direction is None:
            # Initialize linear direction (primarily along corridor Z-axis with some variation)
            self.linear_direction = np.array([
                (np.random.random() - 0.5) * 0.3,  # Small X variation
                (np.random.random() - 0.5) * 0.3,  # Small Y variation
                1.0  # Primary Z direction (along corridor)
            ])
            self.linear_direction = self.linear_direction / np.linalg.norm(self.linear_direction)
        
        # Check if need to reverse (reached end of corridor)
        if position[2] > self.corridor_length * 0.9:
            self.linear_direction[2] = -abs(self.linear_direction[2])
        elif position[2] < self.corridor_length * 0.1:
            self.linear_direction[2] = abs(self.linear_direction[2])
        
        return self.linear_direction * self.max_speed
    
    def _get_circular_velocity(self, position):
        """Circular movement: orbit around a center point."""
        # Update pattern center occasionally to keep movement interesting
        if self.time % 20.0 < 0.1:  # Every 20 seconds
            self.pattern_center = np.array([
                (np.random.random() - 0.5) * self.corridor_width * 0.5,
                self.corridor_height / 2 + (np.random.random() - 0.5) * self.corridor_height * 0.3,
                self.corridor_length / 2 + (np.random.random() - 0.5) * self.corridor_length * 0.3
            ])
        
        # Calculate circular motion in XY plane
        angular_velocity = self.max_speed / self.pattern_radius
        angle = self.pattern_phase + self.time * angular_velocity
        
        # Tangent direction (perpendicular to radius)
        to_center = self.pattern_center - position
        to_center[2] = 0  # Keep circular motion in XY plane
        
        if np.linalg.norm(to_center) > 0.1:
            # Perpendicular vector for circular motion
            tangent = np.array([-to_center[1], to_center[0], 0])
            tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
            
            # Add centering force if too far from pattern center
            distance_from_center = np.linalg.norm(to_center)
            centering_force = to_center / (distance_from_center + 1e-6) * min(distance_from_center / self.pattern_radius, 1.0)
            
            velocity = tangent + centering_force * 0.3
        else:
            velocity = np.array([1, 0, 0])  # Default if at center
        
        # Add slow drift along Z axis
        velocity[2] = np.sin(self.time * 0.2) * 0.3
        
        return velocity * self.max_speed
    
    def _get_zigzag_velocity(self, position):
        """Zigzag movement: oscillate while moving forward."""
        # Primary forward direction along Z
        forward_velocity = self.max_speed * 0.7
        
        # Oscillate in X direction
        oscillation_frequency = 0.5  # Hz
        x_velocity = self.zigzag_amplitude * np.cos(self.time * 2 * np.pi * oscillation_frequency) * oscillation_frequency * 2 * np.pi
        
        # Small oscillation in Y direction (different phase)
        y_velocity = self.zigzag_amplitude * 0.5 * np.sin(self.time * 2 * np.pi * oscillation_frequency * 0.7)
        
        # Reverse direction when reaching corridor ends
        if position[2] > self.corridor_length * 0.9:
            forward_velocity = -abs(forward_velocity)
        elif position[2] < self.corridor_length * 0.1:
            forward_velocity = abs(forward_velocity)
        
        return np.array([x_velocity, y_velocity, forward_velocity])
    
    def _get_random_velocity(self, position):
        """Random wandering movement with goal-seeking."""
        # Move towards goal
        if np.linalg.norm(position - self.goal) < 2.0:
            self.goal = self.generate_random_goal()
        
        to_goal = self.goal - position
        desired_velocity = to_goal / (np.linalg.norm(to_goal) + 1e-6) * self.max_speed
        
        return desired_velocity


class DroneSwarmEnv(gym.Env):
    """Custom Drone Environment with PyBullet physics and swarm obstacles."""
    
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 60}
    
    def __init__(self, corridor_length=100.0, corridor_width=20.0, corridor_height=20.0,
                 buffer_start=0.1, buffer_end=0.9,
                 agent_diameter=0.6, agent_max_speed=10.0, agent_max_acceleration=5.0,
                 num_neighbors=5, num_swarm_drones=10, 
                 swarm_obstacles=None, reward_params=None, lidar_range=10.0, 
                 lidar_rays=36, target_diameter=2.0, max_ep_length=2000, 
                 kinematic_type="holonomic", render_mode=None,
                 # Safety constraint parameters (for Lagrangian RL)
                 safety_enabled=False,
                 collision_weight=0.0,
                 near_miss_weight=0.0,
                 danger_zone_weight=0.0,
                 near_miss_threshold_multiplier=1.5,
                 ignore_walls=False,
                 # Wall proximity parameters (separate from drone proximity)
                 wall_near_miss_weight=None,
                 wall_danger_zone_weight=None,
                 wall_near_miss_threshold=None,
                 # Predicted mode: seed for deterministic random generation
                 predicted_seed=None):
        super().__init__()
        
        self.corridor_length = corridor_length
        self.corridor_width = corridor_width
        self.corridor_height = corridor_height
        self.max_ep_length = max_ep_length
        self.buffer_start = buffer_start
        self.buffer_end = buffer_end
        self.agent_diameter = agent_diameter
        self.agent_max_speed = agent_max_speed
        self.agent_max_acceleration = agent_max_acceleration
        self.num_neighbors = num_neighbors
        self.num_swarm_drones = num_swarm_drones
        self.swarm_obstacles = swarm_obstacles if swarm_obstacles else []
        self.render_mode = render_mode
        self.lidar_range = lidar_range
        self.lidar_rays = lidar_rays
        self.target_diameter = target_diameter
        self.kinematic_type = kinematic_type
        
        # Predicted mode: seed for deterministic random generation
        # When set, random generation uses seed = predicted_seed + episode_count
        self.predicted_seed = predicted_seed
        self.episode_count = 0
        
        # Safety constraint parameters - initialize via configure_safety_constraint
        # Will be set properly below
        self.safety_enabled = False
        self.w_collision_cost = 0.0
        self.w_near_miss_cost = 0.0
        self.w_danger_zone_cost = 0.0
        self.near_miss_threshold = 0.0
        self.ignore_walls = False
        
        # Wall proximity costs (separate from drone proximity)
        self.w_wall_near_miss_cost = 0.0
        self.w_wall_danger_zone_cost = 0.0
        self.wall_near_miss_threshold = 1.0
        
        print(f"\n[DroneSwarmEnv] Kinematic type: {self.kinematic_type}")
        
        # LIDAR configuration - generate rays evenly distributed around the horizontal plane
        # Plus additional rays for up/down coverage
        self.lidar_directions = self._generate_lidar_directions(lidar_rays)
        
        # Reward function parameters
        if reward_params:
            self.w_progress = reward_params['w_progress']
            # Time and jerk should be penalties (negative), but if user provides positive value, convert it
            self.w_time = -abs(reward_params['w_time'])
            self.w_jerk = -abs(reward_params['w_jerk'])
            self.success_reward = abs(reward_params['success_reward'])  # Always positive
            # Crash reward should already be negative from DB, but ensure it is
            crash_val = reward_params['crash_reward']
            self.crash_reward = -abs(crash_val)  # Force negative
            
            print(f"\n[DroneSwarmEnv] Reward function parameters (from config):")
            print(f"  w_progress: {self.w_progress}")
            print(f"  w_time: {self.w_time}")
            print(f"  w_jerk: {self.w_jerk}")
            print(f"  success_reward: {self.success_reward}")
            print(f"  crash_reward: {self.crash_reward}")
        else:
            # This case should not be reached if parameters are passed from the UI
            raise ValueError("reward_params must be provided from the UI")
        
        # Action and observation spaces depend on kinematic type
        if self.kinematic_type == "semi-holonomic":
            # Semi-holonomic: [v_forward, yaw_rate, v_z]
            # v_forward: forward velocity (0 to 1)
            # yaw_rate: rotation rate around z-axis (-1 to 1)
            # v_z: vertical velocity (-1 to 1)
            self.action_space = spaces.Box(
                low=np.array([0.0, -1.0, -1.0]),
                high=np.array([1.0, 1.0, 1.0]),
                dtype=np.float32
            )
            # Observation includes yaw angle
            # - goal_direction (3): signed direction to goal [-1, 1]
            # - goal_distance (1): normalized distance to goal [0, 1]
            # - yaw (1): current yaw angle normalized [-1, 1] (scaled from -pi to pi)
            # - velocity (3): signed normalized current velocity [-1, 1]
            # - lidar_readings (N): normalized distances from LIDAR rays [0, 1]
            # Total: 3 + 1 + 1 + 3 + N
            obs_size = 3 + 1 + 1 + 3 + len(self.lidar_directions)
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(obs_size,),
                dtype=np.float32
            )
            print(f"[DroneSwarmEnv] Semi-holonomic mode - Action: [v_forward, yaw_rate, v_z], Obs size: {obs_size}")
        else:
            # Holonomic: [vx, vy, vz] normalized velocity commands (-1 to 1)
            # -1.0 = full backward/left/down, 0 = stationary, +1.0 = full forward/right/up
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0]),
                high=np.array([1.0, 1.0, 1.0]),
                dtype=np.float32
            )
            # Observation space (all normalized and signed for better learning):
            # - goal_direction (3): signed direction to goal [-1, 1]
            # - goal_distance (1): normalized distance to goal [0, 1]
            # - velocity (3): signed normalized current velocity [-1, 1]
            # - lidar_readings (N): normalized distances from LIDAR rays [0, 1]
            # Total: 3 + 1 + 3 + N (where N = lidar_rays)
            obs_size = 3 + 1 + 3 + len(self.lidar_directions)
            self.observation_space = spaces.Box(
                low=-1.0,  # Allow signed values for directions and velocities
                high=1.0,
                shape=(obs_size,),
                dtype=np.float32
            )
            print(f"[DroneSwarmEnv] Holonomic mode - Action: [vx, vy, vz], Obs size: {obs_size}")
        
        # Physics parameters
        self.max_velocity = agent_max_speed
        
        # Connect to PyBullet - each environment gets its own isolated physics client
        self.physics_client = p.connect(p.DIRECT)
        print(f"[DroneSwarmEnv] Created physics client ID: {self.physics_client}")
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setGravity(0, 0, 0, physicsClientId=self.physics_client)  # No gravity - using direct velocity control
        p.setTimeStep(1/240, physicsClientId=self.physics_client)
        
        self.drone_id = None
        self.swarm_drones = []
        
        # Wall IDs for collision detection
        self.floor_id = None
        self.ceiling_id = None
        self.left_wall_id = None
        self.right_wall_id = None
        self.front_wall_id = None
        self.back_wall_id = None
        self.goal_position = None
        self.start_position = None
        self.previous_distance = None
        self.previous_velocity = None
        self.verbose = 0  # Verbosity level for debugging
        
        # Configure safety constraints with passed parameters
        self.configure_safety_constraint(
            enabled=safety_enabled,
            collision_weight=collision_weight,
            near_miss_weight=near_miss_weight,
            danger_zone_weight=danger_zone_weight,
            near_miss_threshold_multiplier=near_miss_threshold_multiplier,
            ignore_walls=ignore_walls,
            wall_near_miss_weight=wall_near_miss_weight,
            wall_danger_zone_weight=wall_danger_zone_weight,
            wall_near_miss_threshold=wall_near_miss_threshold
        )
    
    def _generate_lidar_directions(self, num_rays):
        """Generate evenly distributed lidar ray directions in 3D space.
        
        Uses Fibonacci sphere algorithm for uniform distribution on a sphere,
        similar to corners/diagonals of a 3D object.
        
        Args:
            num_rays: Number of lidar rays to generate
            
        Returns:
            numpy array of normalized direction vectors
        """
        directions = []
        
        # Use Fibonacci sphere for even distribution in 3D
        # This creates rays pointing in all directions like vertices of a geodesic sphere
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians
        
        for i in range(num_rays):
            # Y coordinate (vertical): ranges from +1 (top) to -1 (bottom)
            y = 1 - (i / float(num_rays - 1)) * 2 if num_rays > 1 else 0
            
            # Radius at this y level
            radius = np.sqrt(1 - y * y)
            
            # Angle around the y-axis
            theta = phi * i
            
            # X and Z coordinates
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            directions.append([x, y, z])
        
        # Normalize all directions (should already be normalized, but ensure it)
        directions = np.array(directions)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.maximum(norms, 1e-6)
        
        return directions
        
    def _create_corridor(self):
        """Create corridor walls using Z as length dimension."""
        # Floor (XZ plane at Y=0)
        floor_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.corridor_width/2, 0.1, self.corridor_length/2],
            physicsClientId=self.physics_client
        )
        self.floor_id = p.createMultiBody(
            baseMass=0, 
            baseCollisionShapeIndex=floor_shape,
            basePosition=[0, -0.1, self.corridor_length/2],
            physicsClientId=self.physics_client
        )
        
        # Ceiling (XZ plane at Y=corridor_height)
        self.ceiling_id = p.createMultiBody(
            baseMass=0, 
            baseCollisionShapeIndex=floor_shape,
            basePosition=[0, self.corridor_height + 0.1, self.corridor_length/2],
            physicsClientId=self.physics_client
        )
        
        # Left and Right walls (YZ planes)
        wall_lr_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.1, self.corridor_height/2, self.corridor_length/2],
            physicsClientId=self.physics_client
        )
        # Left wall (X = -corridor_width/2)
        self.left_wall_id = p.createMultiBody(
            baseMass=0, 
            baseCollisionShapeIndex=wall_lr_shape,
            basePosition=[-self.corridor_width/2 - 0.1, self.corridor_height/2, self.corridor_length/2],
            physicsClientId=self.physics_client
        )
        # Right wall (X = +corridor_width/2)
        self.right_wall_id = p.createMultiBody(
            baseMass=0, 
            baseCollisionShapeIndex=wall_lr_shape,
            basePosition=[self.corridor_width/2 + 0.1, self.corridor_height/2, self.corridor_length/2],
            physicsClientId=self.physics_client
        )
        
        # Front and Back walls (XY planes)
        wall_fb_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.corridor_width/2, self.corridor_height/2, 0.1],
            physicsClientId=self.physics_client
        )
        # Front wall (Z = 0)
        self.front_wall_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_fb_shape,
            basePosition=[0, self.corridor_height/2, -0.1],
            physicsClientId=self.physics_client
        )
        # Back wall (Z = corridor_length)
        self.back_wall_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_fb_shape,
            basePosition=[0, self.corridor_height/2, self.corridor_length + 0.1],
            physicsClientId=self.physics_client
        )
    
    def _create_drone(self, position):
        """Create the agent drone."""
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=self.agent_diameter/2, physicsClientId=self.physics_client)
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE, radius=self.agent_diameter/2, rgbaColor=[0, 1, 1, 1], physicsClientId=self.physics_client
        )
        
        # Initialize with yaw pointing toward goal for semi-holonomic
        if self.kinematic_type == "semi-holonomic" and hasattr(self, 'goal_position'):
            # Calculate initial yaw to point toward goal
            to_goal = self.goal_position - np.array(position)
            initial_yaw = np.arctan2(to_goal[0], to_goal[2])  # Yaw around Y axis (x-z plane)
            # Convert yaw to quaternion
            orientation = p.getQuaternionFromEuler([0, initial_yaw, 0])
        else:
            orientation = [0, 0, 0, 1]  # Default orientation
        
        drone_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation,
            baseInertialFramePosition=[0, 0, 0],
            physicsClientId=self.physics_client
        )
        return drone_id
    
    def configure_safety_constraint(self, enabled, collision_weight, near_miss_weight, 
                                    danger_zone_weight, near_miss_threshold_multiplier, ignore_walls,
                                    wall_near_miss_weight, wall_danger_zone_weight, 
                                    wall_near_miss_threshold):
        """Configure safety constraint parameters for Lagrangian RL.
        
        Args:
            enabled: Whether to enable safety constraint cost computation
            collision_weight: Weight for collision events (per-step, normalized)
            near_miss_weight: Weight for near-miss events (per-step, normalized)
            danger_zone_weight: Weight for being in danger zone (per-step, normalized)
            near_miss_threshold_multiplier: Multiplier for agent diameter to define near-miss threshold
            ignore_walls: If True, walls are completely ignored (no crashes, no proximity costs)
                         If False, wall crashes count as safety violations AND proximity costs apply
            wall_near_miss_weight: Weight for wall near-miss cost (per-step)
            wall_danger_zone_weight: Weight for wall danger zone cost (per-step, additive)
            wall_near_miss_threshold: Distance threshold for wall near-miss in meters
            
        Note: Costs are normalized to be small per-step values (e.g., 0.001-0.01).
              Over a 2000-step episode, total cost will be reasonable for Lagrangian RL.
              Wall costs are ONLY applied when ignore_walls=False.
        """
        self.safety_enabled = enabled
        self.w_collision_cost = collision_weight
        self.w_near_miss_cost = near_miss_weight
        self.w_danger_zone_cost = danger_zone_weight
        self.near_miss_threshold = near_miss_threshold_multiplier * self.agent_diameter
        self.ignore_walls = ignore_walls
        
        # Wall proximity costs (separate from drone proximity)
        self.w_wall_near_miss_cost = wall_near_miss_weight
        self.w_wall_danger_zone_cost = wall_danger_zone_weight
        self.wall_near_miss_threshold = wall_near_miss_threshold
        
        if enabled:
            print(f"\n[DroneSwarmEnv] Safety constraint enabled:")
            print(f"  Collision weight: {collision_weight}")
            print(f"  Near-miss weight (drones): {near_miss_weight}")
            print(f"  Danger zone weight (drones): {danger_zone_weight}")
            print(f"  Near-miss threshold: {self.near_miss_threshold:.2f}m")
            print(f"  Ignore walls for collision cost: {ignore_walls}")
            print(f"  Wall near-miss weight: {wall_near_miss_weight}")
            print(f"  Wall danger zone weight: {wall_danger_zone_weight}")
            print(f"  Wall near-miss threshold: {wall_near_miss_threshold:.2f}m")
    
    def _cast_lidar_rays(self, position):
        """Cast LIDAR rays from drone position to detect obstacles.
        
        Returns:
            list: Distance readings for each ray direction (normalized 0-1)
            list: Hit information [(distance, type, object_id, direction_vector), ...]
                  direction_vector is relative to drone, not world coordinates
        """
        readings = []
        hit_info = []
        
        for direction in self.lidar_directions:
            ray_from = position
            ray_to = position + direction * self.lidar_range
            
            # Cast ray in PyBullet
            result = p.rayTest(ray_from.tolist(), ray_to.tolist(), physicsClientId=self.physics_client)
            
            if result and len(result) > 0:
                hit_obj_id, link_index, hit_fraction, hit_position, hit_normal = result[0]
                
                if hit_obj_id >= 0:  # Hit something
                    distance = hit_fraction * self.lidar_range
                    normalized_distance = distance / self.lidar_range
                    
                    # Determine what was hit
                    hit_type = "unknown"
                    if hit_obj_id == self.drone_id:
                        # Self-hit - treat as no obstacle (max range)
                        hit_type = "self"
                        readings.append(1.0)
                        hit_info.append((self.lidar_range, hit_type, hit_obj_id, direction * self.lidar_range))
                    elif any(hit_obj_id == drone.id for drone in self.swarm_drones):
                        # Hit another drone
                        hit_type = "drone"
                        direction_to_hit = direction * distance
                        readings.append(normalized_distance)
                        hit_info.append((distance, hit_type, hit_obj_id, direction_to_hit))
                    else:
                        # Hit a wall (floor, ceiling, or side walls)
                        hit_type = "wall"
                        direction_to_hit = direction * distance
                        readings.append(normalized_distance)
                        hit_info.append((distance, hit_type, hit_obj_id, direction_to_hit))
                else:
                    # No hit - use max range in this direction
                    readings.append(1.0)
                    hit_info.append((self.lidar_range, "none", -1, direction * self.lidar_range))
            else:
                # No hit - use max range in this direction
                readings.append(1.0)
                hit_info.append((self.lidar_range, "none", -1, direction * self.lidar_range))
        
        return readings, hit_info
    
    def _check_crashes(self, position):
        """Check if drone has crashed into walls or other drones using PyBullet collision detection.
        
        Returns:
            tuple: (crashed: bool, crash_type: str)
        """
        # Check PyBullet contacts
        contact_points = p.getContactPoints(bodyA=self.drone_id, physicsClientId=self.physics_client)
        
        # Wall body IDs
        wall_ids = {self.floor_id, self.ceiling_id, self.left_wall_id, 
                   self.right_wall_id, self.front_wall_id, self.back_wall_id}
        
        if len(contact_points) > 0:
            for contact in contact_points:
                body_b = contact[2]  # Other body ID
                
                # Check if collided with swarm drone
                if any(body_b == drone.id for drone in self.swarm_drones):
                    return True, "drone_collision"
                
                # Check if collided with walls
                if body_b in wall_ids:
                    return True, "wall_collision"
        
        # Additional boundary check (safety fallback)
        safety_margin = self.agent_diameter / 2 + 0.1
        
        # Check corridor boundaries
        if (position[0] < -self.corridor_width/2 + safety_margin or 
            position[0] > self.corridor_width/2 - safety_margin):
            return True, "wall_collision"
        
        if (position[1] < safety_margin or 
            position[1] > self.corridor_height - safety_margin):
            return True, "wall_collision"
        
        if (position[2] < safety_margin or 
            position[2] > self.corridor_length - safety_margin):
            return True, "wall_collision"
        
        return False, "none"
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Apply predicted seed for reproducible scenarios
        # Seed = predicted_seed (timestamp) + episode_count
        if self.predicted_seed is not None:
            episode_seed = self.predicted_seed + self.episode_count
            np.random.seed(episode_seed)
            print(f"[DroneSwarmEnv] Predicted mode: seed={episode_seed} (base={self.predicted_seed}, episode={self.episode_count})")
            self.episode_count += 1
        
        print(f"[DroneSwarmEnv] Resetting environment with physics client ID: {self.physics_client}")
        
        # Clear previous simulation
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, 0, physicsClientId=self.physics_client)  # No gravity - using direct velocity control
        
        # Create corridor
        self._create_corridor()
        
        # Random start position in start zone (buffer_start % of corridor)
        # Account for agent radius to keep it fully inside corridor
        agent_radius = self.agent_diameter / 2
        start_zone_length = self.corridor_length * self.buffer_start
        
        # Ensure start zone is large enough for the drone
        if start_zone_length < 2 * agent_radius + 1:
            print(f"[DroneSwarmEnv] WARNING: Start zone ({start_zone_length}) too small for drone radius ({agent_radius})")
            start_zone_length = max(start_zone_length, 2 * agent_radius + 1)
        
        # Calculate safe spawn ranges with proper margins
        safe_margin = agent_radius + 0.5
        
        # Calculate X range
        x_min = -self.corridor_width/2 + safe_margin
        x_max = self.corridor_width/2 - safe_margin
        x_range = x_max - x_min
        x_pos = x_min + np.random.random() * x_range
        
        # Calculate Y range
        y_min = safe_margin
        y_max = self.corridor_height - safe_margin
        y_range = y_max - y_min
        y_pos = y_min + np.random.random() * y_range
        
        # Calculate Z range
        z_min = safe_margin
        z_max = start_zone_length - safe_margin
        z_range = max(0.1, z_max - z_min)
        z_pos = z_min + np.random.random() * z_range
        
        self.start_position = np.array([x_pos, y_pos, z_pos])
        
        print(f"\n{'='*80}")
        print(f"[DroneSwarmEnv RESET] Corridor dimensions:")
        print(f"  Width: {self.corridor_width}, Height: {self.corridor_height}, Length: {self.corridor_length}")
        print(f"  Agent diameter: {self.agent_diameter}, radius: {agent_radius}, safe_margin: {safe_margin}")
        print(f"[DroneSwarmEnv RESET] Corridor bounds:")
        print(f"  X: [{-self.corridor_width/2:.2f}, {self.corridor_width/2:.2f}]")
        print(f"  Y: [0, {self.corridor_height:.2f}]")
        print(f"  Z: [0, {self.corridor_length:.2f}]")
        print(f"[DroneSwarmEnv RESET] Start zone calculation:")
        print(f"  Buffer start: {self.buffer_start} ({self.buffer_start*100}%)")
        print(f"  Start zone length: {start_zone_length:.2f}")
        print(f"[DroneSwarmEnv RESET] Safe spawn ranges:")
        print(f"  X: [{x_min:.2f}, {x_max:.2f}] (range: {x_range:.2f})")
        print(f"  Y: [{y_min:.2f}, {y_max:.2f}] (range: {y_range:.2f})")
        print(f"  Z: [{z_min:.2f}, {z_max:.2f}] (range: {z_range:.2f})")
        print(f"[DroneSwarmEnv RESET] AGENT SPAWN POSITION: [{x_pos:.2f}, {y_pos:.2f}, {z_pos:.2f}]")
        print(f"  X inside bounds? {x_pos >= -self.corridor_width/2 and x_pos <= self.corridor_width/2}")
        print(f"  Y inside bounds? {y_pos >= 0 and y_pos <= self.corridor_height}")
        print(f"  Z inside bounds? {z_pos >= 0 and z_pos <= self.corridor_length}")
        print(f"{'='*80}\n")
        
        # Random goal position in finish zone (buffer_end to 100%)
        finish_zone_start = self.corridor_length * self.buffer_end
        finish_zone_length = self.corridor_length * (1.0 - self.buffer_end)
        
        # Calculate target bounds
        target_x_min = -self.corridor_width/2 + safe_margin
        target_x_max = self.corridor_width/2 - safe_margin
        target_y_min = safe_margin
        target_y_max = self.corridor_height - safe_margin
        target_z_min = finish_zone_start + safe_margin
        target_z_max = self.corridor_length - safe_margin
        
        self.goal_position = np.array([
            target_x_min + np.random.random() * (target_x_max - target_x_min),
            target_y_min + np.random.random() * (target_y_max - target_y_min),
            target_z_min + np.random.random() * max(0.1, target_z_max - target_z_min)
        ])
        
        print(f"\n[DroneSwarmEnv RESET] Target spawn calculation:")
        print(f"  Finish zone: [{finish_zone_start:.2f}, {self.corridor_length:.2f}] (buffer_end={self.buffer_end})")
        print(f"  Target safe ranges:")
        print(f"    X: [{target_x_min:.2f}, {target_x_max:.2f}]")
        print(f"    Y: [{target_y_min:.2f}, {target_y_max:.2f}]")
        print(f"    Z: [{target_z_min:.2f}, {target_z_max:.2f}]")
        print(f"  TARGET POSITION: [{self.goal_position[0]:.2f}, {self.goal_position[1]:.2f}, {self.goal_position[2]:.2f}]")
        print(f"    X inside bounds? {self.goal_position[0] >= -self.corridor_width/2 and self.goal_position[0] <= self.corridor_width/2}")
        print(f"    Y inside bounds? {self.goal_position[1] >= 0 and self.goal_position[1] <= self.corridor_height}")
        print(f"    Z inside bounds? {self.goal_position[2] >= 0 and self.goal_position[2] <= self.corridor_length}")
        print()
        
        # Create agent drone
        self.drone_id = self._create_drone(self.start_position)
        
        # Verify actual position after creation
        actual_pos, _ = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.physics_client)
        print(f"[DroneSwarmEnv] ACTUAL drone position after creation: {actual_pos}")
        print(f"  Requested: {self.start_position}")
        print(f"  Difference: {np.array(actual_pos) - self.start_position}")
        
        # Create swarm drones in central zone (buffer_start to buffer_end)
        self.swarm_drones = []
        occupied_zones = [self.start_position]
        min_distance = max(3.0, self.agent_diameter * 3)  # Minimum spacing based on agent size
        
        central_zone_start = self.corridor_length * self.buffer_start
        central_zone_length = self.corridor_length * (self.buffer_end - self.buffer_start)
        
        print(f"[DroneSwarmEnv] Creating {self.num_swarm_drones} swarm drones from {len(self.swarm_obstacles)} obstacle configs")
        
        # Create swarm drones based on obstacle configurations
        for i in range(self.num_swarm_drones):
            attempts = 0
            max_attempts = 50
            
            # Get config for this obstacle if available
            config = None
            drone_radius = 0.25  # Default radius
            if i < len(self.swarm_obstacles):
                obstacle = self.swarm_obstacles[i]
                config = {
                    'diameter': obstacle.get('diameter', 0.5),
                    'speed': obstacle.get('speed', 3.0),
                    'acceleration': obstacle.get('acceleration', 1.5),
                    'chaos': obstacle.get('chaos', 0.15),
                    'strategy': obstacle.get('strategy', 'random')
                }
                drone_radius = config['diameter'] / 2
                print(f"[DroneSwarmEnv] Swarm drone {i}: diameter={config['diameter']}, radius={drone_radius}, speed={config['speed']}, accel={config['acceleration']}, chaos={config['chaos']}")
            else:
                print(f"[DroneSwarmEnv] Swarm drone {i}: using defaults (no config available)")
            
            while attempts < max_attempts:
                # Account for drone radius to keep it fully inside corridor
                swarm_margin = drone_radius + 0.5
                position = np.array([
                    -self.corridor_width/2 + swarm_margin + np.random.random() * (self.corridor_width - 2 * swarm_margin),
                    swarm_margin + np.random.random() * max(0.1, self.corridor_height - 2 * swarm_margin),
                    central_zone_start + swarm_margin + np.random.random() * max(0.1, central_zone_length - 2 * swarm_margin)
                ])
                
                # Check collision with occupied zones
                if all(np.linalg.norm(position - oz) >= min_distance for oz in occupied_zones):
                    occupied_zones.append(position)
                    print(f"[DroneSwarmEnv] Swarm drone {i} spawned at position: {position}")
                    print(f"[DroneSwarmEnv]   Valid Z range: [{central_zone_start + drone_radius}, {central_zone_start + central_zone_length - drone_radius}]")
                    drone = SwarmDrone(position.tolist(), self.corridor_length, 
                                      self.corridor_width, self.corridor_height, config=config, 
                                      physics_client=self.physics_client)
                    self.swarm_drones.append(drone)
                    break
                
                attempts += 1
        
        self.step_count = 0
        self.previous_distance = np.linalg.norm(self.start_position - self.goal_position)
        self.previous_velocity = np.zeros(3)
        observation = self._get_observation()
        info = {}
        
        print(f"\n[DroneSwarmEnv RESET COMPLETE]")
        print(f"  self.start_position (drone spawn): {self.start_position}")
        print(f"  self.goal_position (target): {self.goal_position}")
        print(f"  Drone created at Z={self.start_position[2]:.2f}, target at Z={self.goal_position[2]:.2f}\n")
        
        return observation, info
    
    def _get_observation(self):
        """Get current observation vector.
        
        All values are normalized but SIGNED where appropriate for better learning.
        Directions and velocities keep their sign to indicate direction.
        """
        pos, quat = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.physics_client)
        position = np.array(pos)
        vel, ang_vel = p.getBaseVelocity(self.drone_id, physicsClientId=self.physics_client)
        linear_velocity = np.array(vel)
        
        # 1. Goal direction (normalized, SIGNED to preserve direction information)
        goal_vector = self.goal_position - position
        goal_distance_raw = np.linalg.norm(goal_vector)
        goal_direction = goal_vector / (goal_distance_raw + 1e-6)  # Range: [-1, 1]
        
        # 2. Goal distance (normalized by max possible distance in corridor)
        max_possible_distance = np.linalg.norm([self.corridor_length, self.corridor_width, self.corridor_height])
        goal_distance_normalized = np.clip(goal_distance_raw / max_possible_distance, 0, 1)
        
        # 3. Current velocity (normalized by max velocity, SIGNED)
        # Range: [-1, 1] where sign indicates direction
        normalized_velocity = np.clip(
            linear_velocity / self.max_velocity,
            -1.0, 1.0
        )
        
        # 4. LIDAR readings (normalized distances to obstacles - walls and drones)
        lidar_readings, _ = self._cast_lidar_rays(position)
        lidar_normalized = np.array(lidar_readings)  # Already normalized 0-1
        
        # Combine observations based on kinematic type
        if self.kinematic_type == "semi-holonomic":
            # Extract yaw from quaternion
            # PyBullet euler angles are [roll, pitch, yaw] ([x, y, z]), so yaw is index 2.
            euler = p.getEulerFromQuaternion(quat)
            yaw = euler[2]  # Z-axis rotation (yaw around vertical)
            # Normalize yaw from [-pi, pi] to [-1, 1]
            yaw_normalized = yaw / np.pi
            
            # Include yaw in observation
            observation = np.concatenate([
                goal_direction,  # [-1, 1] x 3
                [goal_distance_normalized],  # [0, 1]
                [yaw_normalized],  # [-1, 1]
                normalized_velocity,  # [-1, 1] x 3
                lidar_normalized  # [0, 1] x N
            ]).astype(np.float32)
        else:
            # Holonomic - no yaw needed
            observation = np.concatenate([
                goal_direction,  # [-1, 1] x 3
                [goal_distance_normalized],  # [0, 1]
                normalized_velocity,  # [-1, 1] x 3
                lidar_normalized  # [0, 1] x N
            ]).astype(np.float32)
        
        return observation
    
    def step(self, action):
        """Execute one step in the environment."""
        # Get current position and orientation
        pos, quat = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.physics_client)
        position = np.array(pos)
        
        # Calculate goal direction for debugging
        goal_vector = self.goal_position - position
        goal_distance = np.linalg.norm(goal_vector)
        goal_direction = goal_vector / (goal_distance + 1e-6)
        
        # Apply action based on kinematic type
        if self.kinematic_type == "semi-holonomic":
            # Semi-holonomic: [v_forward, yaw_rate, v_z]
            v_forward = np.clip(action[0], 0, 1)  # Forward velocity (0 to max)
            yaw_rate = np.clip(action[1], -1, 1)  # Yaw rotation rate
            v_z = np.clip(action[2], -1, 1)  # Vertical velocity
            
            # Get current yaw
            # PyBullet euler angles are [roll, pitch, yaw] ([x, y, z]), so yaw is index 2.
            euler = p.getEulerFromQuaternion(quat)
            current_yaw = euler[2]  # Z-axis rotation (yaw)
            
            # Update yaw
            max_yaw_rate = 2.0  # radians per step
            new_yaw = current_yaw + yaw_rate * max_yaw_rate * (1/60)  # Assuming 60 Hz control
            
            # Calculate velocity in world coordinates
            # Forward direction is along z-axis rotated by yaw
            target_velocity = np.array([
                np.sin(new_yaw) * v_forward * self.max_velocity,  # X component
                v_z * self.max_velocity,  # Y component (vertical)
                np.cos(new_yaw) * v_forward * self.max_velocity   # Z component (forward)
            ])
            
            # Update orientation
            new_orientation = p.getQuaternionFromEuler([0, new_yaw, 0])
            p.resetBasePositionAndOrientation(
                self.drone_id, 
                position.tolist(), 
                new_orientation, 
                physicsClientId=self.physics_client
            )
            
            # Debug logging
            if not hasattr(self, 'step_count'):
                self.step_count = 0
            
            if self.step_count % 50 == 0:
                print(f"[DroneEnv Step {self.step_count}] Semi-holonomic mode")
                print(f"  Action: [v_fwd={v_forward:.2f}, yaw_rate={yaw_rate:.2f}, v_z={v_z:.2f}]")
                print(f"  Yaw: {current_yaw:.2f} -> {new_yaw:.2f} rad ({np.degrees(new_yaw):.1f}°)")
                print(f"  Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
                print(f"  Goal: [{self.goal_position[0]:.2f}, {self.goal_position[1]:.2f}, {self.goal_position[2]:.2f}]")
                print(f"  Target velocity: [{target_velocity[0]:.2f}, {target_velocity[1]:.2f}, {target_velocity[2]:.2f}]")
                print(f"  Distance to goal: {goal_distance:.2f}m")
        else:
            # Holonomic: [vx, vy, vz]
            vx = np.clip(action[0], -1, 1)
            vy = np.clip(action[1], -1, 1)
            vz = np.clip(action[2], -1, 1)
            
            # Convert to actual velocity (map -1 to 1 -> -max_velocity to +max_velocity)
            target_velocity = np.array([
                vx * self.max_velocity,
                vy * self.max_velocity,
                vz * self.max_velocity
            ])
            
            # Debug logging
            if not hasattr(self, 'step_count'):
                self.step_count = 0
            
            if self.step_count % 50 == 0:
                print(f"[DroneEnv Step {self.step_count}] Holonomic mode")
                print(f"  Action: [{vx:.2f}, {vy:.2f}, {vz:.2f}]")
                print(f"  Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
                print(f"  Goal: [{self.goal_position[0]:.2f}, {self.goal_position[1]:.2f}, {self.goal_position[2]:.2f}]")
                print(f"  Goal direction: [{goal_direction[0]:.2f}, {goal_direction[1]:.2f}, {goal_direction[2]:.2f}]")
        
        # Apply acceleration limits for more realistic physics
        current_vel, _ = p.getBaseVelocity(self.drone_id, physicsClientId=self.physics_client)
        current_velocity = np.array(current_vel)
        
        # Maximum acceleration per step (prevents instant velocity changes)
        max_accel = self.max_velocity * self.agent_max_acceleration
        velocity_change = target_velocity - current_velocity
        velocity_change_magnitude = np.linalg.norm(velocity_change)
        
        if velocity_change_magnitude > max_accel:
            # Limit acceleration
            velocity_change = (velocity_change / velocity_change_magnitude) * max_accel
        
        new_velocity = current_velocity + velocity_change
        
        # Clamp velocity to max_velocity to prevent exceeding agent max speed
        velocity_magnitude = np.linalg.norm(new_velocity)
        if velocity_magnitude > self.max_velocity:
            new_velocity = (new_velocity / velocity_magnitude) * self.max_velocity
        
        # Set velocity with physics-based constraints
        p.resetBaseVelocity(self.drone_id, linearVelocity=new_velocity.tolist(), physicsClientId=self.physics_client)
        
        # Update swarm drones and step simulation multiple times per action
        # With 240 Hz physics and ~60 Hz control, we need 4 substeps per action
        substeps = 4
        for _ in range(substeps):
            # Update swarm drones
            for drone in self.swarm_drones:
                drone.update(self.swarm_drones, 1/240)
            
            # Step simulation
            p.stepSimulation(physicsClientId=self.physics_client)
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward using custom reward function if provided
        pos, _ = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.physics_client)
        position = np.array(pos)
        vel, ang_vel = p.getBaseVelocity(self.drone_id, physicsClientId=self.physics_client)
        velocity = np.array(vel)
        
        distance_to_goal = np.linalg.norm(position - self.goal_position)
        
        # Cast LIDAR rays
        lidar_readings, lidar_hit_info = self._cast_lidar_rays(position)
        
        # Check success first so goal reach overrides crash detection.
        # Success when agent touches the target (sphere-to-sphere collision)
        # The drone and target are both spheres, so success occurs when:
        # distance_between_centers <= agent_radius + target_radius
        goal_reach_radius = self.target_diameter / 2.0
        agent_radius = self.agent_diameter / 2.0
        combined_reach_radius = goal_reach_radius + agent_radius
        success = distance_to_goal <= combined_reach_radius
        
        # Additional check: Did we pass through the goal during substeps?
        # Check if we were moving toward goal and are now past it
        if not success and self.previous_distance is not None:
            # Check if we crossed the goal sphere during this step
            # If previous distance > radius AND current distance > radius
            # BUT we moved past the goal (dot product indicates we passed it)
            velocity_to_goal = np.dot(velocity, self.goal_position - position)
            if velocity_to_goal < 0 and self.previous_distance < distance_to_goal:
                # We were closer before and now moving away - might have passed through
                # Check if the closest point on our trajectory was within combined reach radius
                if self.previous_distance < combined_reach_radius * 1.5:
                    success = True
                    print(f"[DroneEnv] ✓ TRAJECTORY SUCCESS! Passed through goal (prev_dist={self.previous_distance:.3f}m, curr_dist={distance_to_goal:.3f}m)")
        
        # Check for crashes using improved detection
        # BUT: If we reached the goal, ignore crashes (successful arrival has priority)
        crashed, crash_type = self._check_crashes(position)
        if success:
            crashed = False  # Override crash if goal is reached
            crash_type = "none"
            if self.verbose >= 1:
                print(f"[DroneEnv] ✓ Goal reached - ignoring collision detection (distance={distance_to_goal:.3f}m <= combined_radius={combined_reach_radius:.3f}m)")
        
        # Calculate reward using weight-based approach
        reward = 0.0
        reward_components = {}  # Track individual components for debugging
        
        # Calculate direction to goal for velocity alignment
        goal_vector = self.goal_position - position
        goal_distance_raw = np.linalg.norm(goal_vector)
        goal_direction = goal_vector / (goal_distance_raw + 1e-6)
        
        # 1. Progress reward (distance reduction) - PRIMARY signal
        if self.previous_distance is not None:
            progress = self.previous_distance - distance_to_goal
            progress_reward = self.w_progress * progress
            reward += progress_reward
            reward_components['progress'] = progress_reward
        self.previous_distance = distance_to_goal
        
        # 2. Velocity alignment reward - reward moving TOWARD goal (reduced from 2.0 to 0.2)
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude > 0.1:  # Only if moving
            velocity_direction = velocity / velocity_magnitude
            alignment = np.dot(velocity_direction, goal_direction)  # -1 to 1
            # Reduced scaling factor from 2.0 to 0.2 to prevent reward inflation
            alignment_reward = alignment * 0.2 * (velocity_magnitude / self.max_velocity)
            reward += alignment_reward
            reward_components['velocity_alignment'] = alignment_reward
        else:
            reward_components['velocity_alignment'] = 0.0
        
        # 3. Distance-based potential - REMOVED (caused reward inflation ~+1200 per episode)
        # This exponential reward was duplicating w_progress and causing instability
        
        # 4. Time penalty (encourages faster completion)
        if self.w_time != 0:
            time_penalty = self.w_time
            reward += time_penalty
            reward_components['time'] = time_penalty
        
        # 5. Jerk penalty (encourages smooth movement) - only if enabled
        if self.w_jerk != 0 and self.previous_velocity is not None:
            acceleration = velocity - self.previous_velocity
            jerk = np.linalg.norm(acceleration)
            jerk_penalty = self.w_jerk * jerk
            reward += jerk_penalty
            reward_components['jerk'] = jerk_penalty
        self.previous_velocity = velocity
        
        # 6. Success/crash rewards
        if success:
            # Bonus scales with efficiency (fewer steps = higher bonus)
            # Scale by self.max_ep_length so longer episodes still earn a nonzero bonus.
            efficiency_bonus = max(0, (self.max_ep_length - self.step_count) / self.max_ep_length) * self.success_reward
            total_success_reward = self.success_reward + efficiency_bonus
            reward += total_success_reward
            reward_components['success'] = total_success_reward
            reward_components['efficiency_bonus'] = efficiency_bonus
            print(f"\n{'='*60}")
            print(f"[DroneEnv] ✓ SUCCESS! Agent reached target!")
            print(f"  Steps taken: {self.step_count}")
            print(f"  Final distance: {distance_to_goal:.3f}m")
            print(f"  Combined reach radius: {combined_reach_radius:.3f}m (target={goal_reach_radius:.3f}m + agent={agent_radius:.3f}m)")
            print(f"  Success reward: {self.success_reward:.2f}")
            print(f"  Efficiency bonus: {efficiency_bonus:.2f}")
            print(f"  Total reward: {total_success_reward:.2f}")
            print(f"{'='*60}\n")
        if crashed:
            reward += self.crash_reward  # Should be negative
            reward_components['crash'] = self.crash_reward
            print(f"\n[DroneEnv] ✗ CRASHED! Type: {crash_type}, Penalty: {self.crash_reward:.2f}\n")
        
        # Log reward breakdown every 100 steps for debugging
        if self.step_count % 100 == 0 or crashed or success:
            print(f"[DroneEnv Step {self.step_count}] Reward: {reward_components}, Total: {reward:.2f}, Dist: {distance_to_goal:.2f}m")
        
        # Check termination conditions
        terminated = success or crashed
        
        self.step_count += 1
        # Check if episode has reached max length (truncation)
        truncated = self.step_count >= self.max_ep_length
        if truncated and not terminated:
            # Don't treat timeout as a crash. The agent will learn to avoid it
            # because it doesn't receive the success reward.
            # Timeout truncation is not a safety violation; leave crash_type unchanged.
            print(f"\n[DroneEnv] ⏱ TIMEOUT at {self.step_count} steps (max: {self.max_ep_length})\n")
        
        # Compute safety cost for Lagrangian RL (if enabled)
        cost = 0.0
        if self.safety_enabled:
            # If ignore_walls=True and crash is wall, keep cost at 0.0.
            # This prevents the "suicide loop" where agent stops moving to avoid Lambda.
            # Wall crashes still terminate the episode but don't increase Lambda.
            if self.ignore_walls and crashed and crash_type == 'wall':
                cost = 0.0  # Wall crash doesn't count as safety violation
            else:
                # 1. Collision cost - only for violations (drones when ignore_walls=True, all when False)
                is_safety_violation = crashed
                if self.ignore_walls:
                    # Only drone collisions are violations when ignore_walls=True
                    is_safety_violation = (crashed and crash_type == 'drone')
                
                collision_cost = 1.0 if is_safety_violation else 0.0  # Critical event: 1.0
                
                # 2. Processing Lidar for obstacle proximity
                min_drone_distance_m = float('inf')
                min_wall_distance_m = float('inf')
                
                if len(lidar_hit_info) > 0:
                    for distance_m, hit_type, obj_id, direction in lidar_hit_info:
                        # Separate drone hits from wall hits to apply different proximity costs.
                        if hit_type == "drone":
                            min_drone_distance_m = min(min_drone_distance_m, distance_m)
                        elif hit_type == "wall":
                            min_wall_distance_m = min(min_wall_distance_m, distance_m)
                
                # 3. Drone proximity costs (gradient signal)
                # Small costs for proximity to provide learning gradient
                near_miss_cost = 0.0
                danger_zone_cost = 0.0
                
                if min_drone_distance_m < float('inf'):
                    # Near-miss: gradient cost for proximity warning (0.01 for meaningful signal)
                    if min_drone_distance_m < self.near_miss_threshold:
                        near_miss_cost = 0.01 * self.w_near_miss_cost
                    
                    # Danger zone: higher cost when very close
                    danger_zone_threshold_m = self.near_miss_threshold * 0.5
                    if min_drone_distance_m < danger_zone_threshold_m:
                        danger_zone_cost = 0.01 * self.w_danger_zone_cost
                
                # 4. Wall proximity costs
                # When ignore_walls=True: Wall crashes don't count as violations, but we STILL
                # provide a soft gradient signal to help learning (avoids "dead" wall parameters)
                # When ignore_walls=False: Full wall costs (crashes + proximity)
                wall_near_miss_cost = 0.0
                wall_danger_zone_cost = 0.0
                
                if min_wall_distance_m < float('inf'):
                    if self.ignore_walls:
                        # Soft gradient only - doesn't affect crash violations, just learning signal
                        # Use smaller coefficients to avoid dominating drone avoidance learning
                        if min_wall_distance_m < self.wall_near_miss_threshold:
                            wall_near_miss_cost = 1e-3 * self.w_wall_near_miss_cost  # 10x smaller than drone costs
                        
                        wall_danger_threshold_m = self.wall_near_miss_threshold * 0.5
                        if min_wall_distance_m < wall_danger_threshold_m:
                            wall_danger_zone_cost = 1e-3 * self.w_wall_danger_zone_cost
                    else:
                        # Full wall costs when ignore_walls=False (0.01 for meaningful signal)
                        if min_wall_distance_m < self.wall_near_miss_threshold:
                            wall_near_miss_cost = 0.01 * self.w_wall_near_miss_cost
                        
                        wall_danger_threshold_m = self.wall_near_miss_threshold * 0.5
                        if min_wall_distance_m < wall_danger_threshold_m:
                            wall_danger_zone_cost = 0.01 * self.w_wall_danger_zone_cost
                
                # Total cost: collision (1.0) dominates, proximity adds tiny gradient
                cost = collision_cost + near_miss_cost + danger_zone_cost + wall_near_miss_cost + wall_danger_zone_cost
        
        info = {
            'success': success,
            'crashed': crashed,
            'crash_type': crash_type,
            'distance_to_goal': distance_to_goal,
            'lidar_readings': lidar_readings,
            'lidar_hit_info': lidar_hit_info,
            'truncated': truncated,
            'timeout': truncated and not terminated,
            'cost': float(cost),  # Safety cost for Lagrangian RL (per-step)
            'min_drone_distance': float(min_drone_distance_m) if self.safety_enabled else None,
        }
        
        return observation, reward, terminated, truncated, info
    
    def close(self):
        """Clean up environment."""
        if self.physics_client is not None:
            try:
                # Check if client is still connected before disconnecting
                p.getConnectionInfo(physicsClientId=self.physics_client)
                p.disconnect(self.physics_client)
                print(f"[DroneSwarmEnv] Disconnected physics client {self.physics_client}")
            except Exception as e:
                # Client already disconnected or invalid
                print(f"[DroneSwarmEnv] Physics client {self.physics_client} already disconnected: {e}")
            finally:
                self.physics_client = None
