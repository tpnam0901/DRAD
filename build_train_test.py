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
    brand_folders = os.listdir(data_root)
    brand_folders = [brand for brand in brand_folders if brand.startswith("battery_brand")]
    for index, brand in enumerate(brand_folders):
        if brand.endswith("1"):
            continue
        logging.info(f"Processing brand {index + 1}/{len(brand_folders)}: {brand}")
        prefixs = ["data"] if brand.endswith("3") else ["train", "test"]

        # Find all pkl files in the brand folder and its subfolders
        pkl_files = []
        for prefix in prefixs:
            brand_train_path = os.path.join(data_root, brand, prefix)
            pkl_files += glob.glob(os.path.join(brand_train_path, "*.pkl"))

        all_car_data = {}
        for pkl_file in tqdm(pkl_files):
            metadata = torch.load(pkl_file, weights_only=False)[1]
            filename = os.path.basename(pkl_file)
            # Label is a string "10" or "00". 10 means abnormal, 00 means normal. -> extract the first character
            label = int(metadata["label"][0])
            car_id = metadata["car"]
            mileage = metadata["mileage"]
            charge_segment = metadata["charge_segment"]
            car_data = all_car_data.get(car_id, {})
            car_charge_segment = car_data.get(charge_segment, [])
            car_charge_segment.append((filename, label, car_id, mileage, charge_segment))
            car_data[charge_segment] = car_charge_segment
            all_car_data[car_id] = car_data

        train_data = []
        test_data = []
        for car_id, car_charge_segments in all_car_data.items():
            # Sort charge segments by charge_segment
            sorted_charge_segments = sorted(car_charge_segments.items(), key=lambda x: x[0])
            # Use the first 80% charge segments for training, and the rest for testing
            num_charge_segments = len(sorted_charge_segments)
            num_train_segments = int(num_charge_segments * 0.8)
            for i, (charge_segment, data) in enumerate(sorted_charge_segments):
                if i < num_train_segments:
                    train_data += data
                else:
                    test_data += data

        out_train_path = os.path.join(data_root, brand, "drv_train_labels.csv")
        out_test_path = os.path.join(data_root, brand, f"drv_test_labels.csv")
        with open(out_train_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "label", "car", "mileage", "charge_segment"])
            for filename, label, car_id, mileage, charge_segment in train_data:
                writer.writerow([filename, label, car_id, mileage, charge_segment])
        with open(out_test_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "label", "car", "mileage", "charge_segment"])
            for filename, label, car_id, mileage, charge_segment in test_data:
                writer.writerow([filename, label, car_id, mileage, charge_segment])

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
