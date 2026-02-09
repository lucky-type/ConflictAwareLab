"""Background job processing for ConflictAwareLab experiments."""

from .callback import ExperimentCallback
from .runner import ExperimentRunner

# Export the classes
__all__ = ['ExperimentCallback', 'ExperimentRunner', 'experiment_runner']

# Global experiment runner instance
experiment_runner = ExperimentRunner()
