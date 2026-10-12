import os
import time
from typing import List


def convert_svm_rank_to_libsvm_and_queries(
    svm_rank_input_path: str,
    libsvm_output_path: str,
    query_output_path: str
) -> None:
    """
    Extracts from an SVM-Rank file (with 'qid:X' annotations):
    1. A pure LibSVM file (label feature_id:value ...)
    2. A query file containing the sequential cardinality of each query.
    """
    print(f"Processing file: {svm_rank_input_path}")
    start_time = time.time()
    
    current_qid = None
    current_group_size = 0
    group_sizes: List[int] = []
    
    with open(svm_rank_input_path, 'r', encoding='utf-8') as src_file, \
         open(libsvm_output_path, 'w', encoding='utf-8') as dst_libsvm:
        
        for line in src_file:
            line = line.strip()
            if not line:
                continue
                
            tokens = line.split()
            label = tokens[0]
            qid_token = tokens[1]
            features = tokens[2:]
            
            qid = qid_token.split(':')[1]
            
            # Convert to pure LibSVM format (remove 'qid:X' token)
            clean_libsvm_line = f"{label} " + " ".join(features) + "\n"
            dst_libsvm.write(clean_libsvm_line)
            
            # Count the cardinalities for each group (query)
            if current_qid is None:
                current_qid = qid
                current_group_size = 1
            elif current_qid == qid:
                current_group_size += 1
            else:
                group_sizes.append(current_group_size)
                current_qid = qid
                current_group_size = 1
                
        if current_group_size > 0:
            group_sizes.append(current_group_size)
            
    with open(query_output_path, 'w', encoding='utf-8') as dst_query:
        for size in group_sizes:
            dst_query.write(f"{size}\n")
            
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f}s | Queries extracted: {len(group_sizes)}")


def process_all_folds(base_dataset_dir: str, num_folds: int = 5) -> None:
    """
    Iterates through all folds in the dataset directory and applies the sanitization pipeline.
    """
    for fold_idx in range(1, num_folds + 1):
        fold_dir = os.path.join(base_dataset_dir, f"Fold{fold_idx}")
        
        if not os.path.exists(fold_dir):
            print(f"Warning: Directory {fold_dir} not found. Skipping fold {fold_idx}.")
            continue
            
        print(f"\n--- Starting processing {fold_dir} ---")
        for split in ["train", "vali", "test"]:
            convert_svm_rank_to_libsvm_and_queries(
                svm_rank_input_path=os.path.join(fold_dir, f"{split}.txt"),
                libsvm_output_path=os.path.join(fold_dir, f"{split}_clean.txt"),
                query_output_path=os.path.join(fold_dir, f"{split}_clean.txt.query")
            )


if __name__ == "__main__":
    DATASET_ROOT = "data/MSLR-WEB10K"
    
    print("Starting Data Sanitization pipeline for all folds...")
    global_start = time.time()
    
    process_all_folds(base_dataset_dir=DATASET_ROOT, num_folds=5)
    
    print(f"\nPipeline completed globally in {time.time() - global_start:.2f} seconds.")