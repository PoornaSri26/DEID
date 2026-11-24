#!/usr/bin/env python
"""Download Meddies-PII v2 dataset from Hugging Face."""

from huggingface_hub import snapshot_download
import os
import sys

def download_meddies_pii():
    """Download Meddies-PII v2 dataset."""
    print("=" * 60)
    print("Downloading Meddies-PII v2 dataset from Hugging Face...")
    print("=" * 60)
    
    # Download the dataset
    path = snapshot_download(
        "Meddies/meddies-pii-v2",
        local_dir="./data/meddies-pii"
    )
    
    print(f"\nDataset downloaded to: {path}")
    print("\nAvailable configs:")
    
    # List available configs
    # Just show what's typically available
    print("  - english (English clinical documents)")
    print("  - tamil (Tamil clinical documents)")
    print("  - vietnamese, chinese, japanese, korean, thai")
    print("  - and 12 other languages")
    
    # Verify key files
    train_path = os.path.join(path, "train")
    if os.path.exists(train_path):
        train_files = os.listdir(train_path)
        print(f"\nTraining split has {len(train_files)} files")
    
    print("\n" + "=" * 60)
    print("Download complete! You can now:")
    print("1. Explore the data: python -c \"from datasets import load_dataset; ds = load_dataset('Meddies/meddies-pii-v2', 'english'); print(ds['train'][:2])\"")
    print("2. Start training baselines")
    print("3. Run the hybrid pipeline")
    print("=" * 60)
    
    return path

if __name__ == "__main__":
    path = download_meddies_pii()
    print(f"\nDataset ready at: {path}")
    print("\nNext steps:")
    print("  - Browse: https://huggingface.co/datasets/Meddies/meddies-pii-v2")
    print("  - Check label format in dataset card")
    print("  - Run demo: python src/run_demo.py")