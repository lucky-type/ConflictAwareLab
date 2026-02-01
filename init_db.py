"""Initialize fresh database with all models."""
import sqlalchemy
from app.database import engine, Base
from app.models import (
    Environment, Agent, ResidualConnector, RewardFunction, 
    Experiment, ModelSnapshot, ExperimentMetric #, Curriculum, CurriculumStep
)

def init_db():
    """Create all tables from scratch."""
    print("Creating all database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully!")
    print("\nCreated tables:")
    inspector = sqlalchemy.inspect(engine)
    for table_name in inspector.get_table_names():
        print(f"  - {table_name}")
        columns = [col['name'] for col in inspector.get_columns(table_name)]
        if table_name == "experiments" and "seed" in columns:
             print(f"    ✅ Verified column 'seed' exists in {table_name}")
        elif table_name == "experiments":
             print(f"    ❌ CRITICAL: Column 'seed' MISSING in {table_name}")

if __name__ == "__main__":
    init_db()
