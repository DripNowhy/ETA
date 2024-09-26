import os
import json
import pandas as pd

from dataset.base import BaseDataset


class Attack(BaseDataset):
    def __init__(self, data_root="/home/yiding/Code/VLM-Safety/LLaVA/reject_sampling/data/attack/"):
        super(Attack, self).__init__()
        self.ann_root = data_root

    def get_data(self):
        # Define the path to the CSV file
        path = os.path.join(self.ann_root, "manual_harmful_instructions.csv")
        df = pd.read_csv(path)  # Load the CSV file into a pandas DataFrame

        # Assuming 'instruction' is the column containing harmful instructions
        instructions = df['instruction'].tolist()  # Convert the 'instruction' column to a list

        # Define the common image path for all instructions
        img_path = '/home/yiding/Code/VLM-Safety/LLaVA/reject_sampling/data/attack/prompt_constrained_64.bmp'

        # Create a list of dictionaries where each instruction is associated with the image path
        data = [{'question': instruction, 'image_path': img_path} for instruction in instructions]

        return data
