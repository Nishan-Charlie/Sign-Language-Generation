from datasets import load_dataset
try:
    print("Testing si config...")
    d = load_dataset("polyglots/MADLAD_CulturaX_cleaned", "si", split="train", streaming=True)
    print(next(iter(d)))
except Exception as e:
    print(e)
    try:
        print("Testing default config...")
        d = load_dataset("polyglots/MADLAD_CulturaX_cleaned", split="train", streaming=True)
        print(next(iter(d)))
    except Exception as e2:
        print(e2)
