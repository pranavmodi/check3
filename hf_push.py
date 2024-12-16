from transformers import VisionEncoderDecoderModel
from huggingface_hub import HfApi
import os
import argparse
from pathlib import Path
from huggingface_hub import HfFolder
from dotenv import load_dotenv

def setup_environment():
    """Set up environment variables and HF token."""
    load_dotenv()
    token = os.getenv('HF_TOKEN')
    if not token:
        raise ValueError("Please ensure HF_TOKEN is set in your .env file")
    HfFolder.save_token(token)
    print("✓ Successfully set up Hugging Face token")

def push_latest_checkpoint(checkpoint_dir, hub_model_id, commit_message=None):
    """
    Push the latest checkpoint from the specified directory to Hugging Face Hub.
    
    Args:
        checkpoint_dir (str): Directory containing the checkpoints
        hub_model_id (str): Hugging Face Hub model ID (username/model-name)
        commit_message (str, optional): Custom commit message. Defaults to None.
    """
    print(f"\n📤 Pushing latest checkpoint to Hugging Face Hub ({hub_model_id})...")
    
    # Find the latest checkpoint
    checkpoints = [d for d in os.listdir(checkpoint_dir) 
                  if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir, d))]
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Sort checkpoints by number
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    print(f"Found latest checkpoint: {latest_checkpoint}")
    
    try:
        # Load the model from the checkpoint
        model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)
        
        # Push to hub
        if commit_message is None:
            commit_message = f"Upload model checkpoint: {latest_checkpoint}"
        
        model.push_to_hub(
            hub_model_id,
            commit_message=commit_message,
            private=True  # Set to False if you want a public repository
        )
        print(f"✓ Successfully pushed {latest_checkpoint} to {hub_model_id}")
        
    except Exception as e:
        raise Exception(f"❌ Failed to push to Hugging Face Hub. Error: {str(e)}\n"
                       f"Please check your token and repository permissions.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Push latest checkpoint to Hugging Face Hub')
    parser.add_argument('--checkpoint_dir',
                       type=str,
                       required=True,
                       help='Directory containing the checkpoints')
    parser.add_argument('--hub_model_id',
                       type=str,
                       required=True,
                       help='Hugging Face Hub model ID (username/model-name)')
    parser.add_argument('--commit_message',
                       type=str,
                       help='Custom commit message (optional)')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup environment
    print("\n🔧 Setting up environment...")
    setup_environment()
    
    # Push latest checkpoint
    push_latest_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        hub_model_id=args.hub_model_id,
        commit_message=args.commit_message
    )

if __name__ == "__main__":
    main()
