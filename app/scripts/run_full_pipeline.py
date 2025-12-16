import subprocess
import sys
import time
from pathlib import Path

def run_labeling():
    print("🏷️ Starting Auto-Labeling for missing classes...")
    base_cmd = [sys.executable, "app/scripts/auto_label_dataset.py"]
    base_dir = Path("app/dataset/real")
    
    # Classes to label
    targets = ["broken_streetlight", "dead_animal"]
    
    for target in targets:
        target_dir = base_dir / target / "images"
        if target_dir.exists():
            print(f"   > Labeling {target}...")
            # Blocking call to ensure it finishes before training
            subprocess.run(base_cmd + ["--dir", str(target_dir), "--classes", target], check=False)
        else:
            print(f"   ⚠️ Directory {target_dir} not found.")

def run_training():
    print("\n🏋️ Starting Model Training...")
    subprocess.run([sys.executable, "app/scripts/train_yolo.py"], check=True)

if __name__ == "__main__":
    try:
        run_labeling()
        run_training()
        print("\n✅ pipeline completed successfully!")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
