"""Background job processing for ConflictAwareLab experiments."""

from .callback import ExperimentCallback
from .runner import ExperimentRunner

# Export the classes
__all__ = ['ExperimentCallback', 'ExperimentRunner', 'experiment_runner'] #, 'curriculum_orchestrator']

# Global experiment runner instance
experiment_runner = ExperimentRunner()

# # Import curriculum orchestrator if available
# curriculum_orchestrator = None
# try:
#     from ..orchestrator import CurriculumOrchestrator
#     import asyncio
#     import threading
#     import time

#     # Get the main event loop to pass to orchestrator
#     try:
#         main_loop = asyncio.get_event_loop()
#     except RuntimeError:
#         main_loop = None

#     # Initialize curriculum orchestrator with the main event loop
#     curriculum_orchestrator = CurriculumOrchestrator(
#         experiment_runner=experiment_runner,
#         main_loop=main_loop
#     )

#     # Background thread for curriculum orchestration
#     def run_curriculum_orchestrator_loop():
#         """Background loop that checks and manages curriculum pipelines."""
#         print("[CurriculumOrchestrator] Background thread started")
        
#         while True:
#             try:
#                 curriculum_orchestrator.tick()
#             except Exception as e:
#                 print("[CurriculumOrchestrator] Error in tick: " + str(e))
#                 import traceback
#                 traceback.print_exc()
            
#             # Sleep for 5 seconds between checks
#             time.sleep(5)

#     # Start the orchestrator thread
#     curriculum_thread = threading.Thread(target=run_curriculum_orchestrator_loop, daemon=True, name="CurriculumOrchestrator")
#     curriculum_thread.start()
#     print("[CurriculumOrchestrator] Background thread initialized")

# except Exception as e:
#     print("[CurriculumOrchestrator] Could not initialize: " + str(e))
