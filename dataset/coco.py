import json
import os

from dataset.base import BaseDataset


class CocoDataset(BaseDataset):
    def __init__(self, data_root="/data/coco/"):
        super(CocoDataset, self).__init__()
        self.ann_root = data_root
        self.img_root = data_root

    def get_data(self):
        data = []
        path = os.path.join(self.ann_root, "llava_7b_v1_preference.json")
        # val_phrases = []
        with open(path, "r") as my_file:
            ann = json.load(my_file)
        data = data + [
            {
                "img_path": os.path.join(self.img_root, "train2017/", v['image']),
                "question": v['conversations'][-2]["value"],
                "output1": v['output_1']["value"],
                "output2": v['output_2']["value"],
                "id": v['id'],
                "preference": v["preference"],
                "conversations": v["conversations"]
            } 
            for v in ann
        ]

        return data