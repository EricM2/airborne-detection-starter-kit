import argparse
import copy
import datetime
import fire
import string
import tqdm
import os
from pathlib import Path

from gluoncv.torch.data.gluoncv_motion_dataset.dataset import GluonCVMotionDataset, FieldNames, SplitNames
from gluoncv.torch.data.gluoncv_motion_dataset.utils.ingestion_utils import process_dataset_splits
from gluoncv.torch.data.gluoncv_motion_dataset.utils.serialization_utils import save_json


def ingest_dataset(args, renumber_ids=True):
    """

    :param args: Input arguments
    :param renumber_ids: rename track identities to integers
    """
    dataset = GluonCVMotionDataset(args.anno_name, args.dataset_path, load_anno=False)
    dataset.metadata = {
        FieldNames.DESCRIPTION: "Initial ingestion",
        FieldNames.DATE_MODIFIED: str(datetime.datetime.now()),
    }
    #raw_anno_paths = sorted(Path(dataset.data_root_path).glob("groundtruth.json"))
    raw_anno_paths = sorted(Path('/home/ubuntu/airborne-detection-starter-kit/data/').glob("groundtruth.json"))
    
    for raw_anno_path in tqdm.tqdm(raw_anno_paths):
        # Setting the dataset and samples to None here looks pointless but it allows the memory to be freed, otherwise
        # on subsequent iterations it can actually run out of memory as it loads a new dataset while keeping the
        # previous one still in memory (happened on c5.xlarge 8GB RAM)
        raw_dataset = None
        samples = None
        # raw_sample and sample have references back to the dataset so have to unset these too
        raw_sample = sample = None
        raw_dataset = GluonCVMotionDataset(raw_anno_path)
        raw_dataset.__version__ = 1
        set_dir = raw_anno_path.parent.parent
        images_root_path = Path(dataset.data_root_path) # set_dir / "Images"

        samples = sorted(raw_dataset.samples)
        with open ('/home/ubuntu/siam-mot/data/all_flights_val.txt', 'r') as f:
            all_flights = f.readlines()
        all_flights = [flight.rstrip() for flight in all_flights]

        for raw_id, raw_sample in tqdm.tqdm(samples):
            if raw_id not in all_flights[200:]:
                continue
            data_path = images_root_path /raw_id 
            data_rel_path = str(data_path.relative_to(dataset.data_root_path))
            new_id = data_rel_path
            first_img = sorted(data_path.glob("*.png"))[0]
            first_timestamp = int(first_img.name.split(raw_id)[0])
            sample = raw_sample.get_copy_without_entities(new_id=new_id)
            sample.metadata["orig_path"] = raw_sample.data_relative_path
            sample.data_relative_path = data_rel_path
            unique_ids = {}

            first_frame = None
            for raw_entity in raw_sample.entities:
                entity = copy.deepcopy(raw_entity)
                orig_frame = entity.blob.pop("frame")
                orig_time = entity.time
                if first_frame is None:
                    assert raw_entity.time == first_timestamp
                    first_frame = orig_frame
                rel_frame = orig_frame - first_frame
                # rel_ts = raw_entity.time - first_timestamp
                # assert rel_ts >= 0
                # rel_ts_msec = rel_ts / 1e6
                # ts_msec_round = int(round(rel_ts_msec / sample.period) * sample.period)
                # print(f"frame: {raw_entity.blob.get('frame')} ts_msec: {rel_ts_msec} ts_round {ts_msec_round}")
                # print()
                # assert abs(rel_ts_msec - ts_msec_round) < sample.period / 10
                # entity.time = ts_msec_round

                entity.time = round(rel_frame / sample.fps * 1000)
                if entity.id:
                    obj_type = entity.id.rstrip(string.digits).lower()
                    entity.labels[obj_type] = 1
                    if entity.id.lower() in ("airplane1", "helicopter1"):
                        entity.labels["intruder"] = 1
                    entity.blob["orig_id"] = entity.id
                    if renumber_ids:
                        entity.id = unique_ids.setdefault(entity.id, len(unique_ids))
                entity.blob[FieldNames.FRAME_IDX] = rel_frame
                entity.blob["orig_frame"] = orig_frame
                entity.blob["orig_time"] = orig_time
                if entity.labels and "miss_distance_class" in entity.labels:
                    entity.blob["miss_distance_class"] = entity.labels.pop("miss_distance_class")
                if "range_distance_m" in entity.blob:
                    entity.blob["range_distance_m"] = round(entity.blob["range_distance_m"], 1)
                sample.add_entity(entity)

            # break
            dataset.add_sample(sample, dump_directly=True)

        dataset.dump()

    return dataset


def write_split(dataset):
    def split_func(sample):
        # data_path = sample.data_relative_path
        orig_path = sample.metadata['orig_path']
        if orig_path.startswith("train"):
            return SplitNames.TRAIN
        elif orig_path.startswith("val"):
            return SplitNames.VAL

        raise Exception("Shouldn't happen")

    process_dataset_splits(dataset, split_func, save=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest Prime Air dataset')
    parser.add_argument('--dataset_path', default="/home/ubuntu/airborne-detection-starter-kit/data/val/")
                        #description="The path of dataset folder")
    parser.add_argument('--anno_name', default="anno.json")
                        #description="The file name (with json) of ingested annotation file")
    args = parser.parse_args()

    dataset = ingest_dataset(args, renumber_ids=True)
    write_split(dataset)
