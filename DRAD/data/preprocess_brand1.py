import warnings

# Ignore all warnings
warnings.simplefilter("ignore")
import argparse
import csv
import logging
import os
import random
from glob import glob

import torch
from tqdm.auto import tqdm


def main(data_root: str):
    brand_num = [1]
    for brand in brand_num:
        logging.info(f"Processing brand {brand}/{len(brand_num)}...")
        train_path = os.path.join(data_root, f"battery_brand{brand}", "train")
        test_path = os.path.join(data_root, f"battery_brand{brand}", "test")

        train_pkl_files = glob(os.path.join(train_path, "*.pkl"))
        test_pkl_files = glob(os.path.join(test_path, "*.pkl"))

        if brand == 3:
            train_pkl_files = glob(os.path.join(data_root, f"battery_brand{brand}", "data", "*.pkl"))

        all_car_dict = {}

        for each_path in tqdm(train_pkl_files + test_pkl_files):
            #     print(each_path)

            this_pkl_file = torch.load(each_path)
            this_car_number = this_pkl_file[1]["car"]
            this_car_label = int(this_pkl_file[1]["label"][0])
            this_car_mileage = this_pkl_file[1]["mileage"]
            this_car_charge_segment = this_pkl_file[1]["charge_segment"]
            this_car_filename = os.path.basename(each_path)
            this_car_abs_path = os.path.abspath(each_path)

            if this_car_number not in all_car_dict:
                all_car_dict[this_car_number] = []
                all_car_dict[this_car_number].append(
                    [
                        this_car_number,
                        this_car_label,
                        this_car_mileage,
                        this_car_charge_segment,
                        this_car_filename,
                        this_car_abs_path,
                    ]
                )
            else:
                all_car_dict[this_car_number].append(
                    [
                        this_car_number,
                        this_car_label,
                        this_car_mileage,
                        this_car_charge_segment,
                        this_car_filename,
                        this_car_abs_path,
                    ]
                )
        # shuffle
        random.seed(2025)
        out_train_path = os.path.join(data_root, f"battery_brand{brand}", "train_labels.csv")
        out_test_path = os.path.join(data_root, f"battery_brand{brand}", "test_labels.csv")
        with open(out_train_path, "w", newline="") as train_csvfile, open(out_test_path, "w", newline="") as test_csvfile:
            train_writer = csv.writer(train_csvfile)
            test_writer = csv.writer(test_csvfile)
            train_writer.writerow(["car", "label", "mileage", "charge_segment", "filename", "path"])
            test_writer.writerow(["car", "label", "mileage", "charge_segment", "filename", "path"])
            for car_number, car_data in all_car_dict.items():
                car_data.sort(key=lambda x: x[4])  # Sort by filename (index 4)
                random.shuffle(car_data)
                split_index = int(0.8 * len(car_data))
                train_data = car_data[:split_index]
                test_data = car_data[split_index:]
                for data in train_data:
                    train_writer.writerow(data)
                for data in test_data:
                    test_writer.writerow(data)
        logger.info(f"Preprocessing completed. Train labels saved to {out_train_path}, Test labels saved to {out_test_path}")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dr",
        "--data_root",
        type=str,
        default="battery_data",
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
