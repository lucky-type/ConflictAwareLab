"""Deterministic observation-only heuristic policies for evaluation baselines."""
from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

logger = logging.getLogger(__name__)


def generate_lidar_directions(num_rays: int) -> np.ndarray:
    """Generate lidar ray directions using Fibonacci sphere distribution."""
    if num_rays <= 0:
        raise ValueError("num_rays must be > 0")

    directions = []
    phi = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(num_rays):
        y = 1 - (i / float(num_rays - 1)) * 2 if num_rays > 1 else 0.0
        radius = np.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        directions.append([x, y, z])

    dirs = np.array(directions, dtype=np.float32)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / np.maximum(norms, 1e-6)


@dataclass
class ParsedObservation:
    goal_direction: np.ndarray
    goal_distance: float
    yaw: float | None
    velocity: np.ndarray
    lidar: np.ndarray


class BaseHeuristicPolicy:
    """Base interface for deterministic observation-only heuristic policies."""

    def __init__(self, kinematic_type: str, num_lidar_rays: int):
        if kinematic_type not in {"holonomic", "semi-holonomic"}:
            raise ValueError(f"Unsupported kinematic_type: {kinematic_type}")
        if num_lidar_rays <= 0:
            raise ValueError("num_lidar_rays must be > 0")

        self.kinematic_type = kinematic_type
        self.num_lidar_rays = num_lidar_rays
        self.lidar_directions = generate_lidar_directions(num_lidar_rays)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        parsed = self._parse_observation(obs)
        desired_direction = self._compute_desired_direction(parsed)
        return self._direction_to_action(desired_direction, parsed.yaw)

    def reset(self) -> None:
        """Reset policy state at episode boundaries."""
        return

    def _parse_observation(self, obs: np.ndarray) -> ParsedObservation:
        raw = np.asarray(obs, dtype=np.float32).flatten()

        # AppendLambdaWrapper adds λ as the final observation value.
        expected_without_lambda = (
            3 + 1 + 1 + 3 + self.num_lidar_rays
            if self.kinematic_type == "semi-holonomic"
            else 3 + 1 + 3 + self.num_lidar_rays
        )
        if raw.size == expected_without_lambda + 1:
            raw = raw[:-1]

        if self.kinematic_type == "semi-holonomic":
            goal_direction = raw[0:3]
            goal_distance = float(raw[3]) if raw.size > 3 else 1.0
            yaw_norm = float(raw[4]) if raw.size > 4 else 0.0
            velocity = raw[5:8] if raw.size >= 8 else np.zeros(3, dtype=np.float32)
            lidar = raw[8 : 8 + self.num_lidar_rays]
            yaw = float(np.pi * np.clip(yaw_norm, -1.0, 1.0))
        else:
            goal_direction = raw[0:3]
            goal_distance = float(raw[3]) if raw.size > 3 else 1.0
            velocity = raw[4:7] if raw.size >= 7 else np.zeros(3, dtype=np.float32)
            lidar = raw[7 : 7 + self.num_lidar_rays]
            yaw = None

        if lidar.size < self.num_lidar_rays:
            padded = np.ones(self.num_lidar_rays, dtype=np.float32)
            padded[: lidar.size] = lidar
            lidar = padded

        goal_norm = np.linalg.norm(goal_direction)
        if goal_norm > 1e-6:
            goal_direction = goal_direction / goal_norm
        else:
            goal_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        return ParsedObservation(
            goal_direction=goal_direction.astype(np.float32),
            goal_distance=goal_distance,
            yaw=yaw,
            velocity=np.asarray(velocity, dtype=np.float32),
            lidar=np.asarray(np.clip(lidar, 0.0, 1.0), dtype=np.float32),
        )

    def _compute_desired_direction(self, parsed: ParsedObservation) -> np.ndarray:
        raise NotImplementedError

    def _direction_to_action(self, direction: np.ndarray, yaw: float | None) -> np.ndarray:
        direction = np.asarray(direction, dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        else:
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        if self.kinematic_type == "holonomic":
            return np.clip(direction, -1.0, 1.0).astype(np.float32)

        current_yaw = yaw if yaw is not None else 0.0
        desired_yaw = float(np.arctan2(direction[0], direction[2]))
        yaw_error = _wrap_angle(desired_yaw - current_yaw)
        yaw_rate = np.clip(1.5 * (yaw_error / np.pi), -1.0, 1.0)
        v_forward = np.clip(max(0.0, np.cos(yaw_error)) * 0.9, 0.0, 1.0)
        v_z = np.clip(direction[1], -1.0, 1.0)
        return np.array([v_forward, yaw_rate, v_z], dtype=np.float32)


class PotentialFieldPolicy(BaseHeuristicPolicy):
    """Heuristic policy based on attractive goal and repulsive obstacle fields."""

    def __init__(
        self,
        kinematic_type: str,
        num_lidar_rays: int,
        safety_distance_norm: float = 0.35,
        goal_gain: float = 1.0,
        repulsion_gain: float = 1.6,
    ):
        super().__init__(kinematic_type=kinematic_type, num_lidar_rays=num_lidar_rays)
        self.safety_distance_norm = float(np.clip(safety_distance_norm, 1e-3, 1.0))
        self.goal_gain = goal_gain
        self.repulsion_gain = repulsion_gain

    def _compute_desired_direction(self, parsed: ParsedObservation) -> np.ndarray:
        repulsion = np.zeros(3, dtype=np.float32)
        for idx, distance_norm in enumerate(parsed.lidar):
            if distance_norm < self.safety_distance_norm:
                strength = (self.safety_distance_norm - distance_norm) / self.safety_distance_norm
                repulsion -= self.lidar_directions[idx] * strength

        desired = (self.goal_gain * parsed.goal_direction) + (self.repulsion_gain * repulsion)
        if np.linalg.norm(desired) < 1e-6:
            desired = parsed.goal_direction
        return desired.astype(np.float32)


class VFHLitePolicy(BaseHeuristicPolicy):
    """Vector-Field-Histogram-inspired observation-only navigation baseline."""

    def __init__(
        self,
        kinematic_type: str,
        num_lidar_rays: int,
        blocked_distance_norm: float = 0.40,
        min_free_distance_norm: float = 0.20,
        k_neighbors: int = 6,
        density_smoothing: float = 0.45,
        occupancy_threshold: float = 0.55,
        clearance_weight: float = 0.85,
        goal_weight: float = 1.25,
        density_weight: float = 1.10,
        vertical_penalty: float = 0.18,
        reverse_penalty: float = 0.70,
        inertia_weight: float = 0.25,
        blend_alpha: float = 0.70,
    ):
        super().__init__(kinematic_type=kinematic_type, num_lidar_rays=num_lidar_rays)
        self.blocked_distance_norm = float(np.clip(blocked_distance_norm, 1e-3, 1.0))
        self.min_free_distance_norm = float(np.clip(min_free_distance_norm, 0.0, 1.0))
        self.k_neighbors = max(1, int(k_neighbors))
        self.density_smoothing = float(np.clip(density_smoothing, 0.0, 1.0))
        self.occupancy_threshold = float(np.clip(occupancy_threshold, 0.0, 1.0))
        self.clearance_weight = float(clearance_weight)
        self.goal_weight = float(goal_weight)
        self.density_weight = float(density_weight)
        self.vertical_penalty = float(vertical_penalty)
        self.reverse_penalty = float(reverse_penalty)
        self.inertia_weight = float(inertia_weight)
        self.blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        self.prev_selected_dir: np.ndarray | None = None
        self._neighbor_indices = self._build_neighbor_indices()

    def reset(self) -> None:
        self.prev_selected_dir = None

    def _build_neighbor_indices(self) -> np.ndarray:
        """Precompute nearest neighbors per lidar ray based on cosine similarity."""
        num_rays = self.lidar_directions.shape[0]
        if num_rays == 1:
            return np.zeros((1, 1), dtype=np.int32)

        k = min(self.k_neighbors, num_rays - 1)
        cos_sim = np.matmul(self.lidar_directions, self.lidar_directions.T)
        np.fill_diagonal(cos_sim, -np.inf)
        # Highest cosine means nearest on sphere.
        neighbors = np.argsort(cos_sim, axis=1)[:, -k:]
        return neighbors.astype(np.int32)

    def _compute_smoothed_density(self, lidar: np.ndarray) -> np.ndarray:
        """Estimate obstacle density and smooth over local spherical neighborhood."""
        occupancy = np.clip(
            (self.blocked_distance_norm - lidar) / self.blocked_distance_norm,
            0.0,
            1.0,
        )
        if self._neighbor_indices.size == 0:
            return occupancy.astype(np.float32)

        neighbor_occ_mean = occupancy[self._neighbor_indices].mean(axis=1)
        smoothed = ((1.0 - self.density_smoothing) * occupancy) + (
            self.density_smoothing * neighbor_occ_mean
        )
        return np.clip(smoothed, 0.0, 1.0).astype(np.float32)

    def _compute_desired_direction(self, parsed: ParsedObservation) -> np.ndarray:
        smoothed_density = self._compute_smoothed_density(parsed.lidar)
        free_mask = (
            (parsed.lidar >= self.min_free_distance_norm)
            & (smoothed_density <= self.occupancy_threshold)
        )
        candidate_indices = np.where(free_mask)[0]
        if candidate_indices.size == 0:
            candidate_indices = np.arange(self.num_lidar_rays)

        candidate_dirs = self.lidar_directions[candidate_indices]
        candidate_lidar = parsed.lidar[candidate_indices]
        candidate_density = smoothed_density[candidate_indices]

        goal_alignment = np.einsum("ij,j->i", candidate_dirs, parsed.goal_direction)
        vertical_component = np.abs(candidate_dirs[:, 1])
        reverse_component = np.maximum(0.0, -goal_alignment)

        prev_dir = (
            parsed.goal_direction
            if self.prev_selected_dir is None
            else self.prev_selected_dir
        )
        inertia_alignment = np.einsum("ij,j->i", candidate_dirs, prev_dir)

        scores = (
            (self.clearance_weight * candidate_lidar)
            + (self.goal_weight * goal_alignment)
            - (self.density_weight * candidate_density)
            - (self.vertical_penalty * vertical_component)
            - (self.reverse_penalty * reverse_component)
            + (self.inertia_weight * inertia_alignment)
        )

        best_local_idx = int(np.argmax(scores))
        selected_dir = candidate_dirs[best_local_idx]
        desired = (self.blend_alpha * selected_dir) + (
            (1.0 - self.blend_alpha) * parsed.goal_direction
        )
        desired_norm = np.linalg.norm(desired)
        if desired_norm <= 1e-6:
            desired = parsed.goal_direction
        else:
            desired = desired / desired_norm

        self.prev_selected_dir = desired.astype(np.float32)
        return self.prev_selected_dir


def create_heuristic_policy(
    algorithm: str, kinematic_type: str, num_lidar_rays: int
) -> BaseHeuristicPolicy:
    """Factory for supported deterministic heuristic policies."""
    if algorithm == "potential_field":
        return PotentialFieldPolicy(
            kinematic_type=kinematic_type,
            num_lidar_rays=num_lidar_rays,
            safety_distance_norm=0.35,
            goal_gain=1.0,
            repulsion_gain=1.6,
        )
    if algorithm == "vfh_lite":
        return VFHLitePolicy(
            kinematic_type=kinematic_type,
            num_lidar_rays=num_lidar_rays,
            blocked_distance_norm=0.40,
            min_free_distance_norm=0.20,
            k_neighbors=6,
            density_smoothing=0.45,
            occupancy_threshold=0.55,
            clearance_weight=0.85,
            goal_weight=1.25,
            density_weight=1.10,
            vertical_penalty=0.18,
            reverse_penalty=0.70,
            inertia_weight=0.25,
            blend_alpha=0.70,
        )
    if algorithm == "gap_following":
        logger.warning(
            "Heuristic algorithm 'gap_following' is legacy and mapped to 'vfh_lite'"
        )
        return VFHLitePolicy(
            kinematic_type=kinematic_type,
            num_lidar_rays=num_lidar_rays,
            blocked_distance_norm=0.40,
            min_free_distance_norm=0.20,
            k_neighbors=6,
            density_smoothing=0.45,
            occupancy_threshold=0.55,
            clearance_weight=0.85,
            goal_weight=1.25,
            density_weight=1.10,
            vertical_penalty=0.18,
            reverse_penalty=0.70,
            inertia_weight=0.25,
            blend_alpha=0.70,
        )
    raise ValueError(
        f"Unsupported heuristic algorithm: {algorithm}. "
        f"Supported: ['potential_field', 'vfh_lite']"
    )


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
