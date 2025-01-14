import os
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def load_files_from_directory(directory):
    files_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                files_data[filename] = file.read()
    return files_data

class MemoryStore:
    def __init__(self, memory_directory):
        self.memory_directory = memory_directory
        os.makedirs(self.memory_directory, exist_ok=True)
        self.memory = {}
        self.load_memory()
        print(f"MemoryStore initialized with directory: {self.memory_directory}")

    def add_to_memory(self, key, information):
        self.memory[key] = information
        self.save_memory()
        print(f"Added to memory: {key}")

    def get_from_memory(self, key):
        return self.memory.get(key, None)

    def save_memory(self):
        file_path = os.path.join(self.memory_directory, 'memory_store.json')
        with open(file_path, 'w') as file:
            json.dump(self.memory, file)
        print(f"Memory saved to {file_path}")

    def load_memory(self):
        file_path = os.path.join(self.memory_directory, 'memory_store.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                self.memory = json.load(file)
            print(f"Memory loaded from {file_path}")

    def update_memory_from_directory(self, directory):
        files_data = load_files_from_directory(directory)
        for subject, data in files_data.items():
            self.add_to_memory(subject, data)
        print(f"Memory updated from directory: {directory}")

class Watcher:
    def __init__(self, directory, memory_store):
        self.observer = Observer()
        self.directory = directory
        self.memory_store = memory_store
        print(f"Watcher initialized for directory: {self.directory}")

    def run(self):
        event_handler = Handler(self.memory_store, self.directory)
        self.observer.schedule(event_handler, self.directory, recursive=True)
        self.observer.start()
        print("Watcher started.")

class Handler(FileSystemEventHandler):
    def __init__(self, memory_store, directory):
        self.memory_store = memory_store
        self.directory = directory
        print(f"Handler initialized for directory: {self.directory}")

    def on_modified(self, event):
        if event.is_directory:
            return
        self.memory_store.update_memory_from_directory(self.directory)
        print(f"Directory modified: {self.directory}")

if __name__ == '__main__':
    memory_directory = 'memory'
    watch_directory = 'watch'
    memory_store = MemoryStore(memory_directory)
    watcher = Watcher(watch_directory, memory_store)
    print("Starting watcher...")
    watcher.run()
    print("Watcher started.")
