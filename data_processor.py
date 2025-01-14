import os
import pandas as pd
import numpy as np
from collections import defaultdict, deque

class DynamicDataProcessor:
    def __init__(self, data_file, save_directory=r"E:\Alita\data"):
        self.data_file = data_file
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)
        self.data = pd.read_csv(data_file)
        self.item_counts = defaultdict(int)
        self.queue = deque()
        self.processed_data = None
        print(f"DynamicDataProcessor initialized with data file: {data_file} and save directory: {save_directory}")

    def preprocess_data(self):
        print("Preprocessing data using Pandas and NumPy...")
        self.data.fillna(0, inplace=True)
        self.processed_data = self.data.applymap(np.log1p)
        self.save_data(self.processed_data, "processed_data.csv")
        print("Data preprocessing complete.")

    def count_items(self, items):
        print("Counting items using defaultdict...")
        for item in items:
            self.item_counts[item] += 1
        self.save_data(self.item_counts, "item_counts.json")
        print("Item counting complete.")

    def manage_queue(self, items):
        print("Managing queue using deque...")
        for item in items:
            self.queue.append(item)
        while self.queue:
            processed_item = self.queue.popleft()
            print(f'Processed {processed_item}')
        self.save_data(list(self.queue), "queue.json")
        print("Queue management complete.")

    def save_data(self, data, filename):
        file_path = os.path.join(self.save_directory, filename)
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        elif isinstance(data, defaultdict):
            with open(file_path, 'w') as file:
                json.dump(data, file)
        elif isinstance(data, list):
            with open(file_path, 'w') as file:
                json.dump(data, file)
        else:
            raise ValueError("Unsupported data type for saving.")
        print(f"Data saved to {file_path}")

    def dynamic_process(self, operation_type, *args):
        if operation_type == 'preprocess':
            self.preprocess_data()
        elif operation_type == 'count':
            self.count_items(args[0])
        elif operation_type == 'queue':
            self.manage_queue(args[0])
        else:
            print(f"Unsupported operation type: {operation_type}")

if __name__ == '__main__':
    print("Starting DynamicDataProcessor...")
    processor = DynamicDataProcessor('data.csv')
    processor.dynamic_process('preprocess')
    processor.dynamic_process('count', ['item1', 'item2', 'item3'])
    processor.dynamic_process('queue', ['item1', 'item2', 'item3'])
    print("DynamicDataProcessor finished.")
