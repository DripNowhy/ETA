import json
import os

from dataset.base import BaseDataset


class MMSafetyBench(BaseDataset):
    def __init__(self, data_root="YOUR_DATA_PATH", split="SD_TYPO"):
        super(MMSafetyBench, self).__init__()
        self.split = split
        self.ann_root = data_root
        self.img_root = data_root

    def get_data(self):
        data = []
        # val_phrases = []
        for sc in ['01-Illegal_Activitiy', '02-HateSpeech', '03-Malware_Generation', '04-Physical_Harm',
                     '05-EconomicHarm', '06-Fraud', '07-Sex', '08-Political_Lobbying',
                     '09-Privacy_Violence', '10-Legal_Opinion', '11-Financial_Advice',
                     '12-Health_Consultation', '13-Gov_Decision']:
            path = os.path.join(self.ann_root, f"processed_questions/{sc}.json")
            with open(path, 'r') as my_file:
                ann = json.load(my_file)
            # val_phrases += [ins['Key Phrase'] for ins in ann.values()]
            print(sc)
            data = data + [
                {
                    "image": os.path.join(self.img_root, "imgs/", sc, self.split, f"{k}.jpg"),
                    "question": v['Rephrased Question'],
                    "label": 1,
                    "scenario": f"{sc}",
                    "id": k
                }
                for k, v in ann.items()
            ]

        return data, ["scenario"]