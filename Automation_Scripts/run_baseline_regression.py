import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import and run the main orchestrator
regression_path = os.path.join(
    project_root,
    "CodeBase_Experiments",
    "1_CMAPSS_ML_Degradation_Experiment", 
    "1_Regression_Models"
)
sys.path.insert(0, regression_path)

from run_all_regression import main

if __name__ == "__main__":
    print("Starting Regression Baseline Experiments...")
    print(f"Orchestrator: {regression_path}/run_all_regression.py")
    print()
    main()

