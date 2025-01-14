import os
import json
import shelve
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput

class CharacterManager:
    def __init__(self, character_dropdown, chat_layout, chat_history):
        self.character_dropdown = character_dropdown
        self.chat_layout = chat_layout
        self.chat_history = chat_history
        self.character_name = ""
        self.character_select_btn = None
        print("CharacterManager initialized with empty character name and select button.")

    def on_character_select(self, instance, value):
        self.character_name = value
        self.character_select_btn.text = value
        self.chat_history = self.load_chat_history()
        self.display_greeting()
        # Save the selected character to shelve
        with shelve.open('character_data') as db:
            db['last_character'] = value
        print(f"Character selected: {value}")

    def load_chat_history(self):
        # Dummy implementation, load your chat history as needed
        print("Loading chat history...")
        return ""

    def display_greeting(self):
        greeting_label = self.create_text_input(
            text="{char}: Welcome to the AI Chatbot! How can I assist you today?".replace("{char}", self.character_name)
        )
        self.chat_layout.add_widget(greeting_label)
        print(f"Greeting displayed for character: {self.character_name}")

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
        print(f"Text input created with height: {text_input.height}")
        return text_input

    def load_last_character(self):
        # Load the last selected character from shelve
        with shelve.open('character_data') as db:
            last_character = db.get('last_character')
            if last_character:
                self.character_name = last_character
                self.character_select_btn.text = last_character
                self.display_greeting()
                print(f"Last character loaded: {last_character}")

class CharacterSelectionScreen(Screen):
    def __init__(self, chat_manager, **kwargs):
        super(CharacterSelectionScreen, self).__init__(**kwargs)
        self.chat_manager = chat_manager
        self.layout = BoxLayout(orientation='vertical')
        
        self.scroll_view = ScrollView(size_hint=(1, 1))
        self.grid_layout = GridLayout(cols=1, size_hint_y=None)
        self.grid_layout.bind(minimum_height=self.grid_layout.setter('height'))
        self.scroll_view.add_widget(self.grid_layout)
        
        self.layout.add_widget(self.scroll_view)
        self.add_widget(self.layout)
        
        self.load_characters()
        print("CharacterSelectionScreen initialized and characters loaded.")

    def load_characters(self):
        character_dir = "C:\\Alita\\scripts\\profiles"
        if not os.path.exists(character_dir):
            os.makedirs(character_dir)
            print(f"Character directory created: {character_dir}")
        self.grid_layout.clear_widgets()
        for character_file in os.listdir(character_dir):
            if character_file.endswith(".json"):
                try:
                    with open(os.path.join(character_dir, character_file), "r") as file:
                        character_data = json.load(file)
                        btn = Button(text=character_data["data"]["name"], size_hint_y=None, height=44)
                        btn.bind(on_release=lambda btn, character_data=character_data: self.select_character(character_data))
                        self.grid_layout.add_widget(btn)
                        print(f"Character button added: {character_data['data']['name']}")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {character_file}")

    def select_character(self, character_data):
        self.manager.current = 'chat'
        self.chat_manager.set_character(character_data)
        # Save the selected character to shelve
        with shelve.open('character_data') as db:
            db['last_character'] = character_data["data"]["name"]
        print(f"Character selected and saved: {character_data['data']['name']}")
