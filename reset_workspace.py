import os
import glob

def reset_workspace(base_dir: str = "data/MSLR-WEB10K") -> None:
    """
    Scan all the Fold directories and delete all files generated
    by the data preparation pipeline, restoring the environment to its pristine state.
    """
    print("Starting workspace cleanup...")
    
    # Define the file patterns to delete
    patterns = [
        os.path.join(base_dir, "Fold*", "*_clean.txt"),
        os.path.join(base_dir, "Fold*", "*_clean.txt.query")
    ]
    
    deleted_count = 0
    
    for pattern in patterns:
        files_to_delete = glob.glob(pattern)
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Error while deleting {file_path}: {e}")
                
    print(f"\nCleanup completed! {deleted_count} artifacts deleted. The environment is restored to its pristine state.")

if __name__ == "__main__":
    reset_workspace()