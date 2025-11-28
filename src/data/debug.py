import pandas as pd

df = pd.read_csv("./data/raw/dailydialog/train_daily.csv")

dialog_str = df.iloc[0]['dialog']
print("Raw dialog string:")
print(repr(dialog_str))
print("\n" + "="*80 + "\n")

# Evaluate it
dialog_raw = eval(dialog_str)
print(f"After eval: type={type(dialog_raw)}, len={len(dialog_raw)}")
print("\nFirst element:")
print(repr(dialog_raw[0]))
print("\n" + "="*80 + "\n")

# Try different split methods
print("Split by \\n:")
split1 = dialog_raw[0].split('\n')
print(f"Result: {len(split1)} items")
for i, item in enumerate(split1[:5]):
    print(f"{i}: {repr(item[:50])}")

print("\n" + "="*80 + "\n")

print("Split by ' (single quote at start):")
split2 = [s.strip() for s in dialog_raw[0].split("'") if s.strip() and s.strip() != '\n']
print(f"Result: {len(split2)} items")
for i, item in enumerate(split2[:5]):
    print(f"{i}: {repr(item[:80])}")