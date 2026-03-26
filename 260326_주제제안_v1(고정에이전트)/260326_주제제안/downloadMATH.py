from datasets import load_dataset
import json

ds = load_dataset("nlile/hendrycks-MATH-benchmark")

all_data = []
for split in ds:
    all_data.extend(ds[split].to_list())

with open("MATH.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)