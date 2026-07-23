import os
import pickle
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import argparse

# Same robust patches for legacy pandas pickle loading as in main script
def patch_pandas():
    import sys
    import types
    from pandas import Index, RangeIndex
    
    def patch_module(name):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m
        return sys.modules[name]

    p_indexes = patch_module('pandas.indexes')
    p_indexes_base = patch_module('pandas.indexes.base')
    p_indexes_numeric = patch_module('pandas.indexes.numeric')
    p_indexes_range = patch_module('pandas.indexes.range')

    def _new_Index(cls, d):
        return cls.__new__(cls, **d)

    p_indexes.Index = Index
    p_indexes.base = p_indexes_base
    p_indexes_base.Index = Index
    p_indexes_base._new_Index = _new_Index
    p_indexes_numeric.Int64Index = Index
    p_indexes_numeric.Float64Index = Index
    p_indexes_range.RangeIndex = RangeIndex

    p_core_indexes = patch_module('pandas.core.indexes')
    p_core_indexes_base = patch_module('pandas.core.indexes.base')
    p_core_numeric = patch_module('pandas.core.indexes.numeric')
    p_core_range = patch_module('pandas.core.indexes.range')
    
    p_core_indexes_base._new_Index = _new_Index
    p_core_numeric.Int64Index = Index
    p_core_numeric.Float64Index = Index
    p_core_range.RangeIndex = RangeIndex

CLASS_MAPPING = {
    'Center': 'center',
    'Donut': 'donut',
    'Edge-Loc': 'edge_loc',
    'Edge-Ring': 'edge_ring',
    'Loc': 'loc',
    'Scratch': 'scratch',
    'Random': 'other',
    'Near-full': 'other',
    'none': 'clean'
}

def main():
    parser = argparse.ArgumentParser(description="Convert LSWMD.pkl to PNG images.")
    parser.add_argument("--limit-per-class", type=int, default=100, help="Max images to save per class (default: 100). Set to 0 for all.")
    parser.add_argument("--output-dir", type=str, default="wafer_images", help="Output directory name.")
    args = parser.parse_args()

    pkl_path = os.path.join(os.path.dirname(__file__), "LSWMD.pkl", "LSWMD.pkl")
    if not os.path.exists(pkl_path):
        print(f"Error: Could not find pickle file at {pkl_path}")
        return

    print(f"Loading {pkl_path}...")
    patch_pandas()
    try:
        with open(pkl_path, 'rb') as f:
            df = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Failed to load pickle: {e}")
        return

    print(f"Loaded DataFrame with shape {df.shape}")

    # Extract failure types
    def extract_failure_type(x):
        if isinstance(x, np.ndarray):
            if x.size > 0:
                val = x[0][0]
                return val if isinstance(val, str) else 'none'
        return 'none'

    if 'failureType' in df.columns:
        df['label_raw'] = df['failureType'].apply(extract_failure_type)
    else:
        print("Error: 'failureType' column not found in DataFrame.")
        return

    df['label'] = df['label_raw'].apply(lambda x: CLASS_MAPPING.get(x, 'other')).fillna('clean')

    # Filter valid wafer maps
    def is_valid_wafer(x):
        return isinstance(x, np.ndarray) and x.ndim == 2 and x.size > 0

    df = df[df['waferMap'].apply(is_valid_wafer)].copy()
    print(f"Valid wafer maps: {len(df)}")
    print("Class counts:\n", df['label'].value_counts())

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    classes = df['label'].unique()
    for cls in classes:
        os.makedirs(os.path.join(args.output_dir, cls), exist_ok=True)

    class_counts = {cls: 0 for cls in classes}
    limit = args.limit_per_class if args.limit_per_class > 0 else float('inf')

    print(f"Saving images to '{args.output_dir}' directory (limit: {limit if limit != float('inf') else 'All'} per class)...")
    
    saved_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        label = row['label']
        
        # Check limit
        if class_counts[label] >= limit:
            continue

        wm = row['waferMap'].astype(np.uint8)
        
        # Resize to standard size (224x224)
        resized = cv2.resize(wm, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Min-Max normalize to 0-255
        v_min, v_max = resized.min(), resized.max()
        if v_max - v_min > 0:
            normalized = ((resized - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)
        else:
            normalized = np.zeros((224, 224), dtype=np.uint8)

        # Save image
        save_path = os.path.join(args.output_dir, label, f"{label}_{idx}.png")
        img_pil = Image.fromarray(normalized, mode='L')
        img_pil.save(save_path)
        
        class_counts[label] += 1
        saved_count += 1

    print(f"\nSuccessfully saved {saved_count} images in '{args.output_dir}'.")

if __name__ == "__main__":
    main()
