from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from concurrent.futures import ThreadPoolExecutor
from memory_manager import MemoryStore, load_files_from_directory, Watcher
from machine_learning import MLManager
import spacy
from textblob import TextBlob
import requests

# Load a pre-trained NLP model
nlp = spacy.load("en_core_web_sm")

class ChatManager:
    def __init__(self, screen_manager):
        self.screen_manager = screen_manager
        self.character_dropdown = DropDown()
        self.profile_dropdown = DropDown()
        self.model_buttons = {}
        self.character_name = ""
        self.user_profile = {"name": "You", "description": ""}
        self.chat_history = ""
        self.chat_layout = None
        self.user_name = "You"
        self.character_data = None
        self.profile_manager = None
        self.current_model = ''  # Remove default 'rasa'
        
        # Initialize memory store
        self.memory_store = MemoryStore(r"E:\Alita\Main Memory")
        self.memory_store.load_memory()

        # Load initial files into memory store
        self.memory_store.update_memory_from_directory(r"E:\Alita\Model_Directions")

        # Initialize file watcher
        self.watcher = Watcher(r"E:\Alita\Model_Directions", self.memory_store)
        self.watcher.run()

        # Initialize ML Manager
        self.ml_manager = MLManager()

        print("ChatManager initialized with the following parameters:")
        print(f"Screen Manager: {self.screen_manager}")
        print(f"Character Dropdown: {self.character_dropdown}")
        print(f"Profile Dropdown: {self.profile_dropdown}")
        print(f"Model Buttons: {self.model_buttons}")
        print(f"Character Name: {self.character_name}")
        print(f"User Profile: {self.user_profile}")
        print(f"Chat History: {self.chat_history}")
        print(f"Chat Layout: {self.chat_layout}")
        print(f"User Name: {self.user_name}")
        print(f"Character Data: {self.character_data}")
        print(f"Profile Manager: {self.profile_manager}")
        print(f"Current Model: {self.current_model}")

    def correct_spelling(self, text):
        blob = TextBlob(text)
        corrected_text = str(blob.correct())
        return corrected_text

    def build_chat_interface(self):
        root = BoxLayout(orientation='vertical')

        self.chat_box = ScrollView(size_hint=(1, 0.5), do_scroll_y=True, bar_width=10)
        self.chat_layout = GridLayout(cols=1, size_hint_y=None)
        self.chat_layout.bind(minimum_height=self.chat_layout.setter('height'))
        self.chat_box.add_widget(self.chat_layout)

        self.character_select_btn = Button(text='Select Character', size_hint=(1, 0.1))
        self.character_select_btn.bind(on_release=self.switch_to_character_selection)

        self.profile_select_btn = Button(text='Select Profile', size_hint=(1, 0.1))
        self.profile_select_btn.bind(on_release=self.switch_to_profile_selection)

        self.model_select_btn = Button(text='Select Model', size_hint=(1, 0.1))
        self.model_select_btn.bind(on_release=self.open_model_dropdown)

        self.model_dropdown = DropDown()
        for model_name in list(self.ml_manager.model_paths.keys()):  # Remove 'rasa' from model list
            btn = Button(text=model_name, size_hint_y=None, height=44, background_color=(1, 0, 0, 1))  # Initially red
            btn.bind(on_release=lambda btn: self.switch_model(btn.text))
            self.model_dropdown.add_widget(btn)
            self.model_buttons[model_name] = btn

        self.user_input = TextInput(size_hint=(1, 0.1), multiline=False, foreground_color=(0, 0, 0, 1))
        self.user_input.bind(on_text_validate=self.send_message)

        self.send_button = Button(text="Send", size_hint=(1, 0.1))
        self.send_button.bind(on_press=lambda instance: self.send_message(None))

        self.suggest_button = Button(text="Suggest Response", size_hint=(1, 0.1))
        self.suggest_button.bind(on_press=self.suggest_response)

        root.add_widget(self.character_select_btn)
        root.add_widget(self.profile_select_btn)
        root.add_widget(self.model_select_btn)
        root.add_widget(self.chat_box)
        root.add_widget(self.user_input)
        root.add_widget(self.send_button)
        root.add_widget(self.suggest_button)

        # Initialize character_manager after creating the buttons
        self.character_manager = CharacterManager(self.character_dropdown, self.chat_layout, self.chat_history)
        self.character_manager.character_select_btn = self.character_select_btn

        # Initialize profile_manager after creating the buttons
        self.profile_manager = ProfileManager(self.profile_dropdown, self.screen_manager, self.profile_select_btn)

        self.profile_manager.load_profiles()
        self.profile_manager.load_last_profile()
        self.character_manager.load_last_character()
        self.character_manager.display_greeting()

        # Load models in the background
        self.load_models_in_background()

        return root

    def load_models_in_background(self):
        def update_button_color(model_name, future):
            try:
                future.result()  # Ensure any exceptions are raised
                self.model_buttons[model_name].background_color = (0, 1, 0, 1)  # Change to green when loaded
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")

        with ThreadPoolExecutor() as executor:
            for model_name in self.ml_manager.model_paths:
                future = executor.submit(self.ml_manager.lazy_load_model, model_name)
                future.add_done_callback(lambda future, model_name=model_name: update_button_color(model_name, future))

    def open_model_dropdown(self, instance):
        self.model_dropdown.open(instance)

    def switch_model(self, model_name):
        self.current_model = model_name
        self.model_select_btn.text = f"Model: {model_name}"

    def switch_to_character_selection(self, instance):
        self.screen_manager.current = 'character_selection'

    def switch_to_profile_selection(self, instance):
        self.screen_manager.current = 'profile_selection'

    def send_message(self, instance):
        user_input = self.user_input.text.strip()
        if user_input == '':
            return

        corrected_input = self.correct_spelling(user_input)
        self.display_user_message(corrected_input)
        self.user_input.text = ''
        self.generate_response(corrected_input)
        self.chat_box.scroll_y = 0

    def create_text_input(self, text):
        text_input = TextInput(
            text=text,
            size_hint_y=None,
            readonly=True,
            multiline=True,
            foreground_color=(1, 1, 1, 1),
            background_color=(0, 0, 0, 0),
            disabled_foreground_color=(1, 1, 1, 1),
            padding=(10, 10),
            font_size=14
        )
        token_count = len(text.split())
        text_input.height = max(40, token_count * 2)
        return text_input

    def display_user_message(self, message):
        user_label = self.create_text_input(text=f"[{self.user_profile['name']}]: {message}")
        self.chat_layout.add_widget(user_label)
        self.chat_history += f"\n[{self.user_profile['name']}]: {message}"
        self.save_chat_history()

    def display_ai_message(self, message):
        ai_label = self.create_text_input(text=f"[{self.character_name}]: {message}")
        self.chat_layout.add_widget(ai_label)
        self.chat_history += f"\n[{self.character_name}]: {message}"
        self.save_chat_history()

    def suggest_response(self, instance):
        user_input = self.user_input.text.strip()
        if user_input == '':
            self.display_ai_message("Please enter some text to generate a suggestion.")
            return
        self.generate_response(user_input, suggest=True)

    def generate_response(self, user_input, suggest=False):
        # Fetch guidelines from memory based on current context/subject
        subject = self.identify_subject(user_input)
        guidelines = self.memory_store.get_from_memory(subject)
        if guidelines:
            prompt = f"Lore: {guidelines}\n\n{user_input}"
        else:
            prompt = user_input

        generated_response = self.ml_manager.generate_model_response(prompt, self.current_model)
        if suggest:
            self.display_ai_message(generated_response)
        else:
            self.display_user_message(generated_response)

    def identify_subject(self, user_input):
        doc = nlp(user_input)
        entity_labels = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
        for ent in doc.ents:
            if ent.label_ in entity_labels:
                return ent.text.lower()
        return "default"

    def load_chat_history(self):
        if self.character_name:
            try:
                with open(f"E:\\Alita\\Main Memory\\{self.character_name}_chat_history.txt", "r") as file:
                    return file.read()
            except FileNotFoundError:
                return ""
        return ""

    def save_chat_history(self):
        if self.character_name:
            with open(f"E:\\Alita\\Main Memory\\{self.character_name}_chat_history.txt", "w") as file:
                file.write(self.chat_history)

    def on_profile_select(self, instance, value):
        print(f"Profile selected: {value}")
        self.profile_manager.on_profile_select(value)

    def on_character_select(self, instance, value):
        self.character_manager.on_character_select(instance, value)

    def set_character(self, character_data):
        self.character_data = character_data
        self.character_name = character_data["data"]["name"]
        self.character_select_btn.text = self.character_name
        self.chat_history = self.load_chat_history()  # Load chat history for the selected character
        self.load_chat_history_into_layout()  # Load chat history into the chat layout
        self.character_manager.display_greeting()

    def load_chat_history_into_layout(self):
        self.chat_layout.clear_widgets()
        for line in self.chat_history.split('\n'):
            if line.strip():
                self.chat_layout.add_widget(self.create_text_input(line))

    def save_chat_history(self):
        if self.character_name:
            with open(f"E:\\Alita\\Main Memory\\{self.character_name}_chat_history.txt", "w") as file:
                file.write(self.chat_history)

class MyApp(App):
    def build(self):
        screen_manager = None  # If you're using a screen manager, replace this with the appropriate instance.
        chat_manager = ChatManager(screen_manager)
        return chat_manager.build_chat_interface()

if __name__ == '__main__':
    MyApp().run()
