import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


DATASETS = ("beauty", "sports", "toys", "yelp")
PRETRAIN_DIR = Path("data/pretrain")


def process_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def merge_data(filenames):
    with ProcessPoolExecutor() as executor:
        return [
            item for result in executor.map(process_file, filenames) for item in result
        ]


def write_to_file(data, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        for item in tqdm(data, total=len(data)):
            json.dump(item, file)
            file.write("\n")


def main():
    for split in ("train", "valid"):
        filenames = [
            PRETRAIN_DIR / f"{dataset}_{split}_sampled.jsonl"
            for dataset in DATASETS
        ]
        missing_files = [
            str(filename) for filename in filenames if not filename.is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"Missing {split} inputs: {', '.join(missing_files)}"
            )

        all_data = merge_data(filenames)
        write_to_file(all_data, PRETRAIN_DIR / f"{split}.jsonl")


if __name__ == "__main__":
    main()
