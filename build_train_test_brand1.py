import warnings

# Ignore all warnings
warnings.simplefilter("ignore")

import argparse
import csv
import glob
import logging
import os

import torch
from tqdm.auto import tqdm


def main(data_root: str):
    brand_folders = ["battery_brand1"]
    for index, brand in enumerate(brand_folders):
        logging.info(f"Processing brand {index + 1}/{len(brand_folders)}: {brand}")
        prefixs = ["data"] if brand.endswith("3") else ["train", "test"]
        for prefix in prefixs:
            brand_train_path = os.path.join(data_root, brand, prefix)
            if prefix == "data":
                prefix = "train"
            out_path = os.path.join(data_root, brand, f"drv_{prefix}_labels.csv")
            logging.info(f"Output path for labels: {out_path}")

            pkl_files = glob.glob(os.path.join(brand_train_path, "*.pkl"))
            with open(out_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["filename", "label", "car", "mileage", "charge_segment"])

                for pkl_file in tqdm(pkl_files):
                    metadata = torch.load(pkl_file, weights_only=False)[1]
                    filename = os.path.basename(pkl_file)
                    # Label is a string "10" or "00". 10 means abnormal, 00 means normal. -> extract the first character
                    label = int(metadata["label"][0])
                    car = metadata["car"]
                    mileage = metadata["mileage"]
                    charge_segment = metadata["charge_segment"]
                    writer.writerow([filename, label, car, mileage, charge_segment])

    logging.info("Label files successfully created.")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dr",
        "--data_root",
        type=str,
        default="working/dataset/RFDBattery",
        help="Path to the root directory of the dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()
    logger.setLevel(level)

    main(args.data_root)
