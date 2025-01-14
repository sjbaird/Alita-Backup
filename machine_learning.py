import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor
from flaml import AutoML
from hyperopt import fmin, tpe, hp, Trials
from bayes_opt import BayesianOptimization
import pyswarms as ps
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.clock import Clock

class DynamicDataProcessor:
    def __init__(self, data_file, save_directory=r"E:\Alita\data"):
        self.data_file = data_file
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)
        self.data = pd.read_csv(data_file)
        self.item_counts = defaultdict(int)
        self.queue = deque()
        self.processed_data = None

    def preprocess_data(self):
        print("Preprocessing data using Pandas and NumPy...")
        self.data.fillna(0, inplace=True)
        self.processed_data = self.data.applymap(np.log1p)
        self.save_data(self.processed_data, "processed_data.csv")

    def count_items(self, items):
        print("Counting items using defaultdict...")
        for item in items:
            self.item_counts[item] += 1
        self.save_data(self.item_counts, "item_counts.json")

    def manage_queue(self, items):
        print("Managing queue using deque...")
        for item in items:
            self.queue.append(item)
        while self.queue:
            processed_item = self.queue.popleft()
            print(f'Processed {processed_item}')
        self.save_data(list(self.queue), "queue.json")

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

    def dynamic_process(self, operation_type, *args):
        if operation_type == 'preprocess':
            self.preprocess_data()
        elif operation_type == 'count':
            self.count_items(args[0])
        elif operation_type == 'queue':
            self.manage_queue(args[0])
        else:
            print(f"Unsupported operation type: {operation_type}")

class MLManager:
    def __init__(self, callback=None):
        print("Initializing MLManager...")
        self.models = {}
        self.loaded_models = {}
        self.auto_ml_model = None
        self.model_paths = {
            'codelLlama-13b-hf': os.path.join("E:\\Alita\\models", 'CodeLlama-13b-hf'),
            'codellama-python-13b-hf': os.path.join("E:\\Alita\\models", 'CodeLlama-13b-Python-hf'),
            # Add other models here
        }
        self.save_directory = r"E:\Alita\saved_models"
        self.callback = callback
        self.data_processor = None  # Initialize data processor
        self.initialize_first_model()

    def show_error_popup(self, message):
        def show_popup(dt):
            layout = BoxLayout(orientation='vertical')
            label = Label(text=message)
            close_button = Button(text='Close', size_hint=(1, 0.2))
            layout.add_widget(label)
            layout.add_widget(close_button)
            popup = Popup(title='Error', content=layout, size_hint=(0.8, 0.4))
            close_button.bind(on_release=popup.dismiss)
            popup.open()
        Clock.schedule_once(show_popup, 0)

    def save_model_weights(self, model, model_name):
        print(f"Saving model weights for {model_name}...")
        model_save_path = os.path.join(self.save_directory, f"{model_name}_state.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model weights saved to {model_save_path}")

    def load_model_weights(self, model, model_name):
        print(f"Loading model weights for {model_name}...")
        model_load_path = os.path.join(self.save_directory, f"{model_name}_state.pt")
        if os.path.exists(model_load_path):
            model.load_state_dict(torch.load(model_load_path, map_location='cpu'))
            model.eval()
            print(f"Model weights loaded from {model_load_path}")

    def load_model_and_tokenizer(self, model_path, model_type='causal'):
        print(f"Loading model and tokenizer from {model_path}...")
        if not os.path.exists(model_path):
            self.show_error_popup(f"Model path {model_path} does not exist.")
            return None, None
        try:
            if model_type == 't5':
                tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True, legacy=True)
                model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        except Exception as e:
            self.show_error_popup(f"Failed to load model/tokenizer from {model_path}: {str(e)}")
            print(f"Exception: {e}")
            return None, None
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.save_model_weights(model, model_path.split("\\")[-1])
        self.load_model_weights(model, model_path.split("\\")[-1])
        return tokenizer, model

    def lazy_load_model(self, model_name):
        print(f"Lazy loading model {model_name}...")
        if model_name not in self.loaded_models:
            model_path = self.model_paths[model_name]
            model_type = 't5' if 't5' in model_name else 'causal'
            tokenizer, model = self.load_model_and_tokenizer(model_path, model_type)
            if tokenizer and model:
                self.loaded_models[model_name] = {'tokenizer': tokenizer, 'model': model}
                print(f"Model {model_name} loaded and cached.")
                return True
            else:
                print(f"Failed to load model {model_name}")
                return False
        return True

    def load_models_in_background(self):
        print("Loading remaining models in the background...")
        with ThreadPoolExecutor() as executor:
            futures = {model_name: executor.submit(self.lazy_load_model, model_name) for model_name in self.model_paths if model_name not in self.loaded_models}
            for model_name, future in futures.items():
                try:
                    future.result()
                    print(f"Model {model_name} loaded successfully.")
                except Exception as e:
                    self.show_error_popup(f"Failed to load model {model_name}: {e}")
                    print(f"Exception: {e}")

    def initialize_first_model(self):
        print("Initializing first model...")
        for model_name in self.model_paths:
            if self.lazy_load_model(model_name):
                print(f"First model {model_name} loaded successfully.")
                break
        if self.callback:
            print("Executing callback on the main thread...")
            Clock.schedule_once(lambda dt: self.callback(), 0)

    def check_for_new_or_changed_models(self):
        print("Checking for new or changed models...")
        for model_name, model_path in self.model_paths.items():
            model_save_path = os.path.join(self.save_directory, f"{model_name}_state.pt")
            if not os.path.exists(model_save_path) or os.path.getmtime(model_path) > os.path.getmtime(model_save_path):
                self.lazy_load_model(model_name)

    def generate_model_response(self, prompt, model_name):
        print(f"Generating model response for {model_name}...")
        model_data = self.loaded_models.get(model_name)
        if not model_data:
            self.show_error_popup(f"Model {model_name} is not loaded.")
            return ""
        tokenizer, model = model_data['tokenizer'], model_data['model']
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        attention_mask = torch.ones(inputs.shape, device=model.device)
        max_new_tokens = 100
        temperature = 0.7
        top_k = 50
        top_p = 0.95
        repetition_penalty = 1.2
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def auto_select_model_flaml(self, X_train, y_train):
        print("Auto-selecting model using FLAML...")
        automl = AutoML()
        automl.fit(X_train, y_train, task="classification", time_budget=3600)  # 1 hour time budget
        self.auto_ml_model = automl

    def test_auto_ml(self):
        print("Testing auto ML...")
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        # Load example dataset
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

        # Use FLAML to select the best model
        self.auto_select_model_flaml(X_train, y_train)

        # Make predictions and evaluate the model
        predictions = self.auto_ml_model.predict(X_test)
        accuracy = sum(predictions == y_test) / len(y_test)
        print(f"Test Accuracy: {accuracy}")

    def hyperparameter_tuning(self, X_train, y_train):
        print("Hyperparameter tuning...")
        def objective(params):
            model = SomeMLModel(**params)
            accuracy = cross_val_score(model, X_train, y_train, cv=5).mean()
            return -accuracy

        space = {
            'param1': hp.uniform('param1', 0, 1),
            'param2': hp.uniform('param2', 0, 1)
        }

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
        print(best)

    def bayesian_optimization(self, X_train, y_train):
        print("Bayesian optimization...")
        def black_box_function(param1, param2):
            model = SomeMLModel(param1=param1, param2=param2)
            accuracy = cross_val_score(model, X_train, y_train, cv=5).mean()
            return accuracy

        optimizer = BayesianOptimization(f=black_box_function, pbounds={'param1': (0, 1), 'param2': (0, 1)}, random_state=1)
        optimizer.maximize(init_points=10, n_iter=30)
        print(optimizer.max)

    def swarm_optimization(self, X_train, y_train):
        print("Swarm optimization...")
        def objective_function(params):
            model = SomeMLModel(param1=params[0], param2=params[1])
            accuracy = cross_val_score(model, X_train, y_train, cv=5).mean()
            return -accuracy

        options = {'c1': 0.5, 'c2': 3,'w': 0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
        best_cost, best_pos = optimizer.optimize(objective_function, iters=100)
        print(best_pos)

    def build_transfer_learning_model(self, num_classes):
        print("Building transfer learning model...")
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        
        for layer in base_model.layers:
            layer.trainable = False
        
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def data_augmentation(self, X_train):
        print("Data augmentation...")
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        datagen.fit(X_train)
        return datagen

    def train_transfer_learning_model(self, model, datagen, X_train, y_train, epochs=50):
        print("Training transfer learning model...")
        model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=epochs)

    def solve_optimal_transport(self, cost_matrix, a, b):
        print("Solving optimal transport...")
        import ot
        transport_plan = ot.emd(a, b, cost_matrix)
        return transport_plan

    def feature_engineering(self, df):
        print("Feature engineering...")
        df['feature_1'] = df['column_1'] * df['column_2']
        df['feature_2'] = df['column_3'] ** 2
        return df

    def preprocess_data(self, data_file):
        print(f"Preprocessing data from {data_file}...")
        self.data_processor = DynamicDataProcessor(data_file)
        self.data_processor.preprocess_data()

# Ensure this code only runs when the script is executed directly
if __name__ == "__main__":
    from chat_manager import ChatManager
    from kivy.uix.screenmanager import ScreenManager

    def start_chat_manager(*args):
        print("Starting chat manager...")
        screen_manager = ScreenManager()
        chat_manager = ChatManager(screen_manager)
        chat_manager.build_chat_interface()

    print("Creating MLManager instance...")
    ml_manager = MLManager(callback=start_chat_manager)
    print("Checking for new or changed models...")
    ml_manager.check_for_new_or_changed_models()
    print("Testing auto ML...")
    ml_manager.test_auto_ml()
    print("MLManager setup complete.")
    # Keep the script running
    while True:
        pass
    # Load remaining models in the background
    ml_manager.load_models_in_background()
