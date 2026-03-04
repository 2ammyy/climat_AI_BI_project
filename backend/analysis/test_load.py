# backend/analysis/test_load.py

import pandas as pd

# Try different loading options
print("Trying different loading methods...")

# Method 1: Default
try:
    df1 = pd.read_csv('data/merged_preprocessed.csv')
    print(f"✅ Method 1 (default): {df1.shape}")
except Exception as e:
    print(f"❌ Method 1 failed: {e}")

# Method 2: With explicit delimiter
try:
    df2 = pd.read_csv('data/merged_preprocessed.csv', delimiter=',')
    print(f"✅ Method 2 (comma): {df2.shape}")
except Exception as e:
    print(f"❌ Method 2 failed: {e}")

# Method 3: Skip bad lines
try:
    df3 = pd.read_csv('data/merged_preprocessed.csv', on_bad_lines='skip')
    print(f"✅ Method 3 (skip bad): {df3.shape}")
except Exception as e:
    print(f"❌ Method 3 failed: {e}")

# Method 4: Check encoding
try:
    df4 = pd.read_csv('data/merged_preprocessed.csv', encoding='utf-8')
    print(f"✅ Method 4 (utf-8): {df4.shape}")
except Exception as e:
    print(f"❌ Method 4 failed: {e}")