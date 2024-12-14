import os
import subprocess
import shutil
from pathlib import Path

def setup_sroie_dataset(data_dir="data"):
    """
    Downloads and sets up the ICDAR-2019-SROIE dataset.
    
    Args:
        data_dir (str): Directory where the dataset should be stored
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Check if data directory already exists and has content
        if os.path.exists(data_dir) and os.path.exists(os.path.join(data_dir, "img")) and os.path.exists(os.path.join(data_dir, "key")):
            print(f"Dataset already exists in {data_dir}")
            return True
            
        # Create data directory if it doesn't exist
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        
        # Clone the repository
        subprocess.run([
            "git", "clone", 
            "https://github.com/zzzDavid/ICDAR-2019-SROIE.git"
        ], check=True)
        
        # Remove existing data directory if it exists
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        
        # Copy data from cloned repository
        shutil.copytree("ICDAR-2019-SROIE/data", data_dir)
        
        # Clean up
        shutil.rmtree("ICDAR-2019-SROIE")
        if os.path.exists(os.path.join(data_dir, "box")):
            shutil.rmtree(os.path.join(data_dir, "box"))
            
        print(f"Successfully set up SROIE dataset in {data_dir}")
        return True
        
    except Exception as e:
        print(f"Error setting up dataset: {str(e)}")
        return False

if __name__ == "__main__":
    # Set default data directory
    data_directory = "data"
    
    # Run the setup function
    success = setup_sroie_dataset(data_directory)
    
    if success:
        print("Dataset setup completed successfully")
    else:
        print("Dataset setup failed")
