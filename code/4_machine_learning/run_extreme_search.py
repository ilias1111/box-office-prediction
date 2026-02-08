import os
from datetime import datetime
from ml_logic_extreme import MOTR

if __name__ == "__main__":
    # --- CONFIGURATION ---
    GRID_TYPE = "random_search"
    ID_COLUMN_NAME = "movie_id"
    
    # EXTREME SETTINGS
    # On a 3090/4090, 200 iterations per model is feasible for tabular data.
    # Adjust N_ITER based on time budget.
    N_ITER = 200 
    CV_FOLDS = 5
    
    # PARALLELISM
    # Set to 1 or 2 to avoid OOM on GPU.
    # If using a single 24GB GPU, N_PARALLEL_JOBS=1 or 2 is safe.
    # If using multiple GPUs, increase this.
    N_PARALLEL_JOBS = 1 
    
    # Calculate absolute path to the data directory (2 levels up from this script)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'ml_ready_data')
    
    DATA_FILES_LIST = os.listdir(DATA_DIR)
    
    # You can filter here if you want to run only specific datasets
    # DATA_FILES_LIST = [f for f in DATA_FILES_LIST if "binary_classification" in f]

    TASK_TYPE_LIST = [i.split("__")[1] for i in DATA_FILES_LIST]
    TARGET_COLUMN_NAME_LIST = [
        "revenue_usd_adj" if i == "regression" else i for i in TASK_TYPE_LIST
    ]

    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    cases = len(
        [
            (i, j, k)
            for i, j, k in zip(DATA_FILES_LIST, TASK_TYPE_LIST, TARGET_COLUMN_NAME_LIST)
        ]
    )
    
    print(f"Starting EXTREME run {RUN_ID} with {cases} datasets.")
    print(f"Iterations: {N_ITER}, CV Folds: {CV_FOLDS}, Parallel Jobs: {N_PARALLEL_JOBS}")
    
    counter = 1
    for data_file, task_type, target_column_name in zip(
        DATA_FILES_LIST, TASK_TYPE_LIST, TARGET_COLUMN_NAME_LIST
    ):
        print(f"\n[{counter}/{cases}] Processing {data_file}...")
        counter += 1
        
        trainer = MOTR(
            RUN_ID,
            os.path.join(DATA_DIR, data_file),
            target_column_name,
            ID_COLUMN_NAME,
            task_type=task_type,
            grid_type=GRID_TYPE,
            positive_class="Success",
            cv_folds=CV_FOLDS,
            random_search_iter=N_ITER,
            search_n_jobs=1, # Keep internal search jobs low to prioritize GPU usage
            model_n_jobs=-1, # Use all available cores/GPU for the model itself
            enable_pipeline_cache=False, 
        )
        
        # We use run() with minimal parallelism to avoid VRAM exhaustion
        trainer.run(parallel=True, n_parallel_jobs=N_PARALLEL_JOBS)
        
    print(f"\nEXTREME run {RUN_ID} completed.")
