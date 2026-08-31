from datasets import load_dataset

ds = load_dataset("Leoxx/whowhen_pro", split="text")
ds.save_to_disk("./whowhen_pro_text")