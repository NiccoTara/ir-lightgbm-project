import time
import lightgbm as lgb
import numpy as np

def load_fold_datasets(fold_num: int, enable_efb: bool):
    """
    Factory function to load the datasets of a specific Fold.
    Encapsulates the complexity of the paths and the .query files.
    """
    base_dir = f"data/MSLR-WEB10K/Fold{fold_num}"

    # parameters for enabling/disabling EFB at the root level
    dataset_params = {"enable_bundle": enable_efb}
    
    # internal helper function to read the query files
    def load_groups(filepath):
        with open(filepath, 'r') as f:
            return np.array([int(line.strip()) for line in f])
            
    train_group = load_groups(f"{base_dir}/train_clean.txt.query")
    valid_group = load_groups(f"{base_dir}/vali_clean.txt.query")
    test_group = load_groups(f"{base_dir}/test_clean.txt.query")

    train_data = lgb.Dataset(f"{base_dir}/train_clean.txt", group=train_group, params=dataset_params)
    valid_data = lgb.Dataset(f"{base_dir}/vali_clean.txt", reference=train_data, group=valid_group, params=dataset_params)
    test_data = lgb.Dataset(f"{base_dir}/test_clean.txt", reference=train_data, group=test_group, params=dataset_params)
    
    # force loading into memory and EFB compression (if enabled)
    train_data.construct()  # force loading into memory and EFB compression (if enabled)
    
    return train_data, valid_data, test_data

class TimeTrackingCallback:
    """
    Custom LightGBM callback to record the cumulative elapsed time 
    at the end of each boosting iteration.
    """
    def __init__(self):
        self.start_time = None
        self.time_history = []

    def __call__(self, env):
        # env is a lightgbm.callback.CallbackEnv object
        if self.start_time is None:
            # Initialize the start time at the very first iteration
            self.start_time = time.time()
        
        # Calculate the elapsed time since the beginning of training
        elapsed = time.time() - self.start_time
        
        # Append the cumulative time to our history list
        self.time_history.append(elapsed)

def run_experiment(conf_name: str, use_goss: bool, use_efb: bool, return_dict):
    fold = 1
    
    print(f"\n{'='*50}")
    print(f"Start experiment: {conf_name}")
    print(f"{'='*50}")
    
    train_data, valid_data, test_data = load_fold_datasets(fold_num=fold, enable_efb=use_efb)
    
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "learning_rate": 0.05,
        "min_child_weight": 100,
        "num_leaves": 255,
        "verbose": -1,
        "seed": 42,
        "num_threads": 1
    }
    
    if use_goss:
        params["data_sample_strategy"] = "goss"
        params["top_rate"] = 0.1
        params["other_rate"] = 0.1
    else:
        params["data_sample_strategy"] = "bagging"
        params["bagging_fraction"] = 1.0
        params["bagging_freq"] = 0

    evals_result = {}
    start_time = time.time()

    time_tracker = TimeTrackingCallback()

    model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=1000,
        valid_sets=[valid_data, test_data],
        valid_names=["valid", "test"],
        callbacks=[
            lgb.log_evaluation(period=200),
            lgb.record_evaluation(evals_result),
            time_tracker
        ]
    )
    
    elapsed_time = time.time() - start_time
    final_ndcg = evals_result['test']['ndcg@10'][-1]
    
    print(f"\n[Result] {conf_name}]")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Final NDCG@10:      {final_ndcg:.4f}\n")

    return_dict[conf_name] = {
        "time": elapsed_time,
        "ndcg_history": evals_result['test']['ndcg@10'],
        "time_history": time_tracker.time_history
    }