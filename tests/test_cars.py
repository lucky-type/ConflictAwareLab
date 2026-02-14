"""
Automated tests for CARS (Confidence-Adaptive Risk Shielding) mechanism.

These tests validate the core CARS logic, ablation toggles, metric consistency,
and determinism guarantees required by the dissertation experiment plan v3.

Run: source venv/bin/activate && pytest tests/test_cars.py -v
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock heavy dependencies BEFORE importing app modules.
# This allows tests to run without stable_baselines3, pybullet, torch, etc.
# ---------------------------------------------------------------------------

# Create mock modules for heavy dependencies
_mock_sb3 = types.ModuleType("stable_baselines3")
for cls_name in ("PPO", "A2C", "SAC", "TD3", "DDPG"):
    setattr(_mock_sb3, cls_name, MagicMock())
sys.modules["stable_baselines3"] = _mock_sb3
sys.modules["stable_baselines3.common.callbacks"] = types.ModuleType("stable_baselines3.common.callbacks")
sys.modules["stable_baselines3.common.callbacks"].BaseCallback = MagicMock
sys.modules["stable_baselines3.common.vec_env"] = types.ModuleType("stable_baselines3.common.vec_env")
sys.modules["stable_baselines3.common.monitor"] = types.ModuleType("stable_baselines3.common.monitor")

_mock_pybullet = types.ModuleType("pybullet")
sys.modules["pybullet"] = _mock_pybullet

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Helpers: lightweight mock environment (no PyBullet dependency)
# ---------------------------------------------------------------------------

class DummyDroneEnv(gym.Env):
    """Minimal 3-DOF environment for unit-testing wrappers."""

    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int = 10, act_dim: int = 3, fixed_obs: np.ndarray | None = None):
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
        self._fixed_obs = fixed_obs
        self._step_count = 0

    def reset(self, **kwargs):
        self._step_count = 0
        obs = self._fixed_obs if self._fixed_obs is not None else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = self._fixed_obs if self._fixed_obs is not None else np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        done = self._step_count >= 10
        info = {"cost": 0.0, "success": done, "crashed": False}
        return obs, reward, done, False, info


class DummyBaseModel:
    """Mock base model that returns a fixed deterministic action."""

    def __init__(self, action: np.ndarray, obs_dim: int = 10):
        self._action = np.asarray(action, dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)

    def predict(self, obs, deterministic=True):
        return self._action.copy(), None


class DummyLagrangeState:
    """Minimal LagrangeState replacement."""

    def __init__(self, lam: float = 0.0):
        self.lam = lam


# ---------------------------------------------------------------------------
# Fixture: create a ResidualActionWrapper with the given params
# ---------------------------------------------------------------------------

def _make_wrapper(
    base_action: np.ndarray = np.array([0.5, 0.0, 0.5]),
    k_factor: float = 0.15,
    adaptive_k: bool = False,
    enable_k_conf: bool = True,
    enable_k_risk: bool = True,
    lam: float = 0.0,
    obs_dim: int = 10,
):
    """Build a ResidualActionWrapper with a mocked base model (no file I/O)."""
    from app.wrappers import ResidualActionWrapper

    env = DummyDroneEnv(obs_dim=obs_dim)
    lagrange = DummyLagrangeState(lam=lam)

    wrapper = ResidualActionWrapper.__new__(ResidualActionWrapper)
    # Manually initialise required attributes instead of calling __init__
    # (which would try to load a real model from disk).
    gym.Wrapper.__init__(wrapper, env)

    wrapper.base_model = DummyBaseModel(action=base_action, obs_dim=obs_dim)
    wrapper.base_expects_lambda = False
    wrapper.base_model_path = "mock"
    wrapper.base_algorithm = "PPO"
    wrapper.lagrange_state = lagrange
    wrapper.base_lambda_frozen = 0.0
    wrapper.k_factor = k_factor
    wrapper.k_min = 0.01
    wrapper.k_max = 1.0
    wrapper.adaptive_k = adaptive_k
    wrapper.k_min_ratio = 0.7
    wrapper.conflict_tau = -0.3
    wrapper.conflict_gamma = 2.0
    wrapper.k_conf_ema_alpha = 0.05
    wrapper.risk_alpha = 0.05
    wrapper.risk_boost_max = 0.5
    wrapper.progress_boost = 1.2
    wrapper.enable_k_conf = enable_k_conf
    wrapper.enable_k_risk = enable_k_risk
    wrapper._ema_k_conf = 1.0
    wrapper._effective_k_sum = 0.0
    wrapper._effective_k_count = 0
    wrapper._last_effective_k = k_factor
    wrapper._last_cos_sim = 0.0
    wrapper._last_K_conf = 1.0
    wrapper._last_K_risk = 1.0
    wrapper._last_K_progress = 1.0
    wrapper.last_base_action = None
    wrapper.last_residual_action = None
    wrapper.last_final_action = None

    # Reset to initialise _last_obs
    wrapper.reset()
    return wrapper


# ===========================================================================
# TEST 1: CARS-Null == Static-K (sanity check)
# ===========================================================================

class TestCARSNullSanity:
    """M10 (CARS-Null) must produce identical actions to Static-K with same k_factor."""

    def test_cars_null_equals_static(self):
        base_action = np.array([0.6, -0.2, 0.4], dtype=np.float32)
        residual = np.array([0.1, 0.3, -0.1], dtype=np.float32)
        k = 0.15

        w_static = _make_wrapper(base_action=base_action, k_factor=k, adaptive_k=False)
        _, _, _, _, info_s = w_static.step(residual)

        w_null = _make_wrapper(
            base_action=base_action, k_factor=k,
            adaptive_k=True, enable_k_conf=False, enable_k_risk=False,
        )
        _, _, _, _, info_n = w_null.step(residual)

        static_action = info_s["residual_info"]["final_action"]
        null_action = info_n["residual_info"]["final_action"]
        np.testing.assert_allclose(static_action, null_action, atol=1e-6,
                                   err_msg="CARS-Null action differs from Static-K")
        assert abs(info_n["residual_info"]["effective_k"] - k) < 1e-6

    def test_cars_null_k_components_are_one(self):
        w = _make_wrapper(adaptive_k=True, enable_k_conf=False, enable_k_risk=False)
        residual = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        _, _, _, _, info = w.step(residual)
        assert info["residual_info"]["K_conf"] == 1.0
        assert info["residual_info"]["K_risk"] == 1.0


# ===========================================================================
# TEST 2: EMA reset between episodes
# ===========================================================================

class TestEMAReset:

    def test_ema_resets_on_episode_boundary(self):
        w = _make_wrapper(
            base_action=np.array([1.0, 0.0, 0.0]),
            adaptive_k=True, enable_k_conf=True, enable_k_risk=False,
            k_factor=0.15,
        )
        opposing = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(20):
            w.step(opposing)
        assert w._ema_k_conf != 1.0, "EMA did not change during episode"

        w.reset()
        assert w._ema_k_conf == 1.0, "EMA was not reset between episodes"

        _, _, _, _, info = w.step(opposing)
        k_conf_first_step = info["residual_info"]["K_conf"]
        assert abs(k_conf_first_step - 1.0) < 0.5, (
            f"K_conf on first step after reset is {k_conf_first_step}, expected near 1.0"
        )


# ===========================================================================
# TEST 3: K_conf toggle
# ===========================================================================

class TestKConfToggle:

    def test_k_conf_disabled_stays_one(self):
        w = _make_wrapper(adaptive_k=True, enable_k_conf=False, enable_k_risk=True, lam=5.0)
        opposing = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(10):
            _, _, _, _, info = w.step(opposing)
            assert info["residual_info"]["K_conf"] == 1.0


# ===========================================================================
# TEST 4: K_risk toggle
# ===========================================================================

class TestKRiskToggle:

    def test_k_risk_disabled_stays_one(self):
        w = _make_wrapper(adaptive_k=True, enable_k_conf=True, enable_k_risk=False, lam=100.0)
        residual = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        for _ in range(5):
            _, _, _, _, info = w.step(residual)
            assert info["residual_info"]["K_risk"] == 1.0

    def test_k_risk_increases_with_lambda(self):
        w = _make_wrapper(adaptive_k=True, enable_k_conf=False, enable_k_risk=True, lam=10.0)
        residual = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        _, _, _, _, info = w.step(residual)
        k_risk = info["residual_info"]["K_risk"]
        assert k_risk > 1.0, f"K_risk should be > 1.0, got {k_risk}"
        assert k_risk <= 1.5, f"K_risk should be <= 1.5, got {k_risk}"


# ===========================================================================
# TEST 5: Action clipping detection
# ===========================================================================

class TestActionClipping:

    def test_clipping_detected_when_saturated(self):
        w = _make_wrapper(base_action=np.array([0.8, 0.0, 0.0]), k_factor=0.5, adaptive_k=False)
        big_residual = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = w.step(big_residual)
        assert info["residual_info"]["action_clipped"] is True

    def test_no_clipping_when_within_bounds(self):
        w = _make_wrapper(base_action=np.array([0.1, 0.0, 0.0]), k_factor=0.05, adaptive_k=False)
        small = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = w.step(small)
        assert info["residual_info"]["action_clipped"] is False


# ===========================================================================
# TEST 6: Violation rate consistency
# ===========================================================================

class TestViolationRateConsistency:

    def test_violation_rate_same_logic(self):
        from app.background_jobs.evaluation import EvaluationMixin
        evaluator = EvaluationMixin()
        costs = [0.05, 0.15, 0.30, 0.50, 0.08]
        epsilon = 0.1
        metrics = evaluator._calculate_evaluation_metrics(
            episode_rewards=[1.0]*5, episode_lengths=[100]*5,
            episode_successes=[True]*5, episode_crashes=[False]*5,
            episode_timeouts=[False]*5, episode_costs=costs,
            episode_near_misses=[0]*5, episode_danger_time=[0]*5,
            lagrange_state=None, safety_config={"enabled": True, "risk_budget": epsilon},
        )
        expected = sum(c > epsilon for c in costs) / len(costs) * 100.0
        assert abs(metrics["violation_rate"] - expected) < 1e-6

    def test_violation_not_normalized_by_length(self):
        from app.background_jobs.evaluation import EvaluationMixin
        evaluator = EvaluationMixin()
        metrics = evaluator._calculate_evaluation_metrics(
            episode_rewards=[1.0], episode_lengths=[1000],
            episode_successes=[True], episode_crashes=[False],
            episode_timeouts=[False], episode_costs=[0.5],
            episode_near_misses=[0], episode_danger_time=[0],
            lagrange_state=None, safety_config={"enabled": True, "risk_budget": 0.1},
        )
        assert metrics["violation_rate"] == 100.0


# ===========================================================================
# TEST 7: Raw metrics stored (unrounded)
# ===========================================================================

class TestRawMetrics:

    def test_raw_metrics_present_in_evaluation(self):
        from app.background_jobs.evaluation import EvaluationMixin
        evaluator = EvaluationMixin()
        successes = [True, False, True, False, True, False, False]
        metrics = evaluator._calculate_evaluation_metrics(
            episode_rewards=[1.0]*7, episode_lengths=[100]*7,
            episode_successes=successes, episode_crashes=[not s for s in successes],
            episode_timeouts=[False]*7, episode_costs=[0.0]*7,
            episode_near_misses=[0]*7, episode_danger_time=[0]*7,
            lagrange_state=None, safety_config=None,
        )
        expected_raw = 3.0 / 7.0 * 100.0
        assert "success_rate_raw" in metrics
        assert abs(metrics["success_rate_raw"] - expected_raw) < 1e-10
        assert metrics["success_rate"] != metrics["success_rate_raw"]


# ===========================================================================
# TEST 8: CARS formula correctness
# ===========================================================================

class TestCARSFormula:

    def test_effective_k_within_bounds(self):
        w = _make_wrapper(adaptive_k=True, enable_k_conf=True, enable_k_risk=True, k_factor=0.15, lam=100.0)
        opposing = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(50):
            _, _, _, _, info = w.step(opposing)
            ek = info["residual_info"]["effective_k"]
            assert w.k_min <= ek <= w.k_max

    def test_static_k_is_constant(self):
        k = 0.20
        w = _make_wrapper(k_factor=k, adaptive_k=False)
        residual = np.array([0.3, -0.5, 0.1], dtype=np.float32)
        for _ in range(10):
            _, _, _, _, info = w.step(residual)
            assert info["residual_info"]["effective_k"] == k


# ===========================================================================
# TEST 9: Cosine similarity computation
# ===========================================================================

class TestCosineSimilarity:

    def test_aligned_actions_positive_cos_sim(self):
        w = _make_wrapper(base_action=np.array([1.0, 0.0, 0.0]))
        _, _, _, _, info = w.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert info["residual_info"]["cos_sim"] > 0.9

    def test_opposing_actions_negative_cos_sim(self):
        w = _make_wrapper(base_action=np.array([1.0, 0.0, 0.0]))
        _, _, _, _, info = w.step(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
        assert info["residual_info"]["cos_sim"] < -0.9

    def test_zero_residual_cos_sim_zero(self):
        w = _make_wrapper(base_action=np.array([1.0, 0.0, 0.0]))
        _, _, _, _, info = w.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert info["residual_info"]["cos_sim"] == 0.0


# ===========================================================================
# TEST 10: Action composition formula
# ===========================================================================

class TestActionComposition:

    def test_action_composition_correctness(self):
        base = np.array([0.3, -0.2, 0.5], dtype=np.float32)
        residual = np.array([0.4, 0.6, -0.3], dtype=np.float32)
        k = 0.2
        w = _make_wrapper(base_action=base, k_factor=k, adaptive_k=False)
        _, _, _, _, info = w.step(residual)
        expected = np.clip(base + k * residual, -1.0, 1.0)
        actual = np.array(info["residual_info"]["final_action"])
        np.testing.assert_allclose(actual, expected, atol=1e-6)


# ===========================================================================
# TEST 11: Analysis script statistical functions
# ===========================================================================

class TestAnalysisStatistics:
    """Validate core statistical functions from analysis/run_analysis.py."""

    def test_paired_permutation_test_identical(self):
        from analysis.run_analysis import paired_permutation_test
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        diff, p = paired_permutation_test(x, y, alternative="two-sided")
        assert diff == 0.0
        assert p == 1.0

    def test_paired_permutation_test_clear_difference(self):
        from analysis.run_analysis import paired_permutation_test
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        diff, p = paired_permutation_test(x, y, alternative="greater")
        assert diff > 0
        assert p < 0.05  # Should be significant

    def test_bootstrap_ci_contains_mean(self):
        from analysis.run_analysis import bootstrap_ci
        x = np.array([10.0, 12.0, 11.0, 13.0, 9.0, 11.5, 10.5, 12.5])
        mean, lo, hi = bootstrap_ci(x)
        assert lo <= mean <= hi
        assert lo < hi

    def test_cohens_d_zero_for_identical(self):
        from analysis.run_analysis import cohens_d
        x = np.array([1.0, 2.0, 3.0])
        d = cohens_d(x, x)
        assert abs(d) < 1e-10

    def test_benjamini_hochberg_correction(self):
        from analysis.run_analysis import benjamini_hochberg
        p_values = [0.01, 0.04, 0.03, 0.20]
        results = benjamini_hochberg(p_values, alpha=0.05)
        # First three should be significant after BH, last one not
        assert results[0][2] is True   # p=0.01 → significant
        assert results[3][2] is False  # p=0.20 → not significant
