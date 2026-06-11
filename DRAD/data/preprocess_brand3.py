import glob
import math
import os
import os.path as osp

import numpy as np
import torch
from tqdm.auto import tqdm

column = torch.load("battery_data/battery_brand3/column.pkl")
data_files = glob.glob("battery_data/battery_brand3/data/*.pkl")

car_data = {}

for file in tqdm(data_files):
    data, meta_data = torch.load(file)
    car_id = meta_data["car"]
    segments = car_data.get(car_id, [])
    segments.append((data, meta_data))
    car_data[car_id] = segments

# Checking all segments for each car have the same car id
for car_id, segments in car_data.items():
    assert len(segments) > 0
    for data, meta_data in segments:
        assert car_id == meta_data["car"]

out_dir = "battery_data/battery_brand3/data_by_segments/"
os.makedirs(out_dir, exist_ok=True)
normal_car = set()
abnormal_car = set()
normal_lengths = []
abnormal_lengths = []
for car_id, segments in tqdm(car_data.items()):
    car_charge_seg = {}
    for data, meta_data in segments:
        charge_segment = meta_data["charge_segment"]
        car_seg = car_charge_seg.get(charge_segment, [])
        car_seg.append((data, meta_data))
        car_charge_seg[charge_segment] = car_seg
    for charge_segment in car_charge_seg:
        # Merge all segments by timestamp
        all_data = []
        segment_ids = set()
        meta_data = car_charge_seg[charge_segment][0][1]
        for data, cur_meta_data in car_charge_seg[charge_segment]:
            all_data.append(data)
            segment_ids.add(cur_meta_data["charge_segment"])
            for k, v in cur_meta_data.items():
                if not meta_data[k] == v:
                    # There is some mismatch due to float precision, in mileage
                    print(meta_data)
                    print(cur_meta_data)
                    print(
                        f"[WARNING] Meta data mismatch for car {car_id} charge segment {charge_segment} on key {k}. Value 1: {meta_data[k]}, Value 2: {v}"
                    )

        assert len(segment_ids) == 1

        # sort by timestamp
        all_data = sorted(all_data, key=lambda x: x[0, column.index("timestamp")])
        all_data = np.concatenate(all_data, axis=0)
        # Check the next timestamp is always larger than the previous one
        for i in range(1, len(all_data)):
            assert all_data[i, column.index("timestamp")] > all_data[i - 1, column.index("timestamp")]

        meta_data["label"] = meta_data["label"][0]
        if int(meta_data["label"]) == 0:
            normal_car.add(car_id)
            normal_lengths.append(len(all_data))
        else:
            abnormal_car.add(car_id)
            abnormal_lengths.append(len(all_data))
        out_file = f"{out_dir}/{car_id}_{charge_segment}.pkl"
        torch.save((all_data, meta_data), out_file)

print(f"Normal cars: {len(normal_car)}, Abnormal cars: {len(abnormal_car)}")
print(f"Normal lengths: mean={np.mean(normal_lengths)}, std={np.std(normal_lengths)}")
print(f"Abnormal lengths: mean={np.mean(abnormal_lengths)}, std={np.std(abnormal_lengths)}")

# Five-fold cross validation split
normal_car = list(normal_car)
abnormal_car = list(abnormal_car)
# For each fold, we use 20% of normal segments and 20% of abnormal segments in each car for testing. The rest is for training.
num_folds = 5
all_cars = normal_car + abnormal_car
assert len(all_cars) == 100
print("Starting cross-validation split...")
for fold in range(num_folds):
    fold_ratio_start = fold * 1 / num_folds
    fold_ratio_end = (fold + 1) * 1 / num_folds
    with open(f"battery_data/battery_brand3/fold_{fold}_train.txt", "w") as train_file, open(
        f"battery_data/battery_brand3/fold_{fold}_test.txt", "w"
    ) as test_file:
        for car_id in all_cars:
            car_segments = glob.glob(f"{out_dir}/{car_id}_*.pkl")
            assert len(car_segments) > 0
            num_segments = len(car_segments)
            car_segments = sorted(car_segments)
            val_start_idx = math.floor(fold_ratio_start * num_segments)
            val_end_idx = math.ceil(fold_ratio_end * num_segments)
            val_segments = car_segments[val_start_idx:val_end_idx]
            train_segments = car_segments[:val_start_idx] + car_segments[val_end_idx:]
            assert len(val_segments) + len(train_segments) == num_segments
            assert len(set(val_segments).intersection(set(train_segments))) <= 1
            assert len(val_segments) > 0
            assert len(train_segments) > 0
            assert len(train_segments) > len(val_segments)
            for seg in train_segments:
                train_file.write(f"{osp.basename(seg)}\n")
            for seg in val_segments:
                test_file.write(f"{osp.basename(seg)}\n")
print("Cross-validation split done.")
