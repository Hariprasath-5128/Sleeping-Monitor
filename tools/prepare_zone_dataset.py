import os
import glob
import shutil
import random
import sys

# Import thermal_utils from the user's files
sys.path.insert(0, r"C:\Projects\sleeping-monitor\backend")
from thermal_utils import process_image, classify_zone

INPUT_DIR = r"C:\Projects\sleeping-monitor\data\new_dataset_IR"
OUTPUT_DIR = r"C:\Projects\sleeping-monitor\data\labeled_zone_dataset"

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    sequences = sorted([d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))])
    if not sequences:
        print("No sequences found in", INPUT_DIR)
        return
        
    # Shuffle and split 80/20 by sequence (so same sequence doesn't leak into test)
    random.seed(42)
    random.shuffle(sequences)
    
    split_idx = int(len(sequences) * 0.8)
    train_seqs = sequences[:split_idx]
    test_seqs = sequences[split_idx:]
    
    print(f"Total sequences: {len(sequences)}")
    print(f"Train sequences: {len(train_seqs)}")
    print(f"Test sequences: {len(test_seqs)}")
    
    counts = {"train": {"LEFT": 0, "RIGHT": 0, "CENTER": 0}, "test": {"LEFT": 0, "RIGHT": 0, "CENTER": 0}}
    
    for split, seq_list in [("train", train_seqs), ("test", test_seqs)]:
        print(f"Processing {split} split...")
        for zone in ["LEFT", "RIGHT", "CENTER"]:
            os.makedirs(os.path.join(OUTPUT_DIR, split, zone), exist_ok=True)
            
        for seq in seq_list:
            seq_dir = os.path.join(INPUT_DIR, seq)
            frames = glob.glob(os.path.join(seq_dir, "**", "*.png"), recursive=True)
            for frame in frames:
                # Use the flawless rule-based math to generate the Ground Truth label
                feats, mask, thresh, plausible = process_image(frame)
                if feats is None or not plausible:
                    continue # Skip if person isn't visible enough
                
                zone = classify_zone(feats["centroid_x_frac"])
                
                # Copy file to its labeled directory
                filename = f"{seq}_{os.path.basename(frame)}"
                dest = os.path.join(OUTPUT_DIR, split, zone, filename)
                shutil.copy2(frame, dest)
                counts[split][zone] += 1
                
    print("\nDataset preparation complete!")
    print(f"Train counts: {counts['train']}")
    print(f"Test counts: {counts['test']}")

if __name__ == "__main__":
    main()
