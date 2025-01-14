import os
import json
import shelve
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput

class ProfileManager:
    def __init__(self, profile_dropdown, screen_manager, profile_select_btn):
        print("Initializing ProfileManager...")
        self.profile_dropdown = profile_dropdown
        self.screen_manager = screen_manager
        self.user_profile = {"name": "You", "description": ""}
        self.user_name = ""
        self.profile_select_btn = profile_select_btn

    def load_profiles(self):
        print("Loading profiles...")
        profile_dir = "E:\\Alita\\Profiles\\User"
        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir)
        self.profile_dropdown.clear_widgets()
        for profile_file in os.listdir(profile_dir):
            if profile_file.endswith(".json"):
                try:
                    with open(os.path.join(profile_dir, profile_file), "r") as file:
                        profile_data = json.load(file)
                        btn = Button(text=profile_data["name"], size_hint_y=None, height=44)
                        btn.bind(on_release=lambda btn, profile_data=profile_data: self.on_profile_select(profile_data))
                        self.profile_dropdown.add_widget(btn)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {profile_file}")

    def load_last_profile(self):
        print("Loading last profile...")
        # Load the last selected profile from shelve
        with shelve.open('profile_data') as db:
            last_profile = db.get('last_profile')
            if last_profile:
                print(f"Loading last profile: {last_profile}")
                self.set_profile(last_profile)

    def delete_profile(self, profile_name):
        print(f"Deleting profile: {profile_name}")
        profile_dir = "E:\\Alita\\Profiles\\User"
        profile_file = os.path.join(profile_dir, f"{profile_name}.json")
        if os.path.exists(profile_file):
            os.remove(profile_file)
            self.load_profiles()

    def on_profile_select(self, profile_data):
        print(f"Selected profile: {profile_data}")
        self.set_profile(profile_data)
        # Save the selected profile to shelve
        with shelve.open('profile_data') as db:
            db['last_profile'] = profile_data

    def set_profile(self, profile_data):
        print(f"Setting profile: {profile_data}")
        self.user_profile = profile_data
        self.user_name = profile_data["name"]
        self.profile_select_btn.text = f'{profile_data["name"]}: {profile_data["description"]}'
        # Save the selected profile to shelve
        with shelve.open('profile_data') as db:
            db['last_profile'] = profile_data

class ProfileSelectionScreen(Screen):
    def __init__(self, profile_manager, **kwargs):
        super(ProfileSelectionScreen, self).__init__(**kwargs)
        print("Initializing ProfileSelectionScreen...")
        self.profile_manager = profile_manager
        self.layout = BoxLayout(orientation='vertical')
        
        self.scroll_view = ScrollView(size_hint=(1, 0.6))
        self.grid_layout = GridLayout(cols=1, size_hint_y=None)
        self.grid_layout.bind(minimum_height=self.grid_layout.setter('height'))
        self.scroll_view.add_widget(self.grid_layout)
        
        self.user_name_input = TextInput(hint_text='Enter your name', size_hint=(1, 0.1))
        self.user_description_input = TextInput(hint_text='Enter your description', size_hint=(1, 0.2))
        self.save_button = Button(text='Save Profile', size_hint=(1, 0.1))
        self.save_button.bind(on_press=self.save_profile)
        
        self.delete_profile_button = Button(text='Delete Profile', size_hint=(1, 0.1))
        self.delete_profile_button.bind(on_press=self.delete_profile)
        
        self.chat_button = Button(text='Chat', size_hint=(1, 0.1))
        self.chat_button.bind(on_press=self.switch_to_chat)
        
        self.layout.add_widget(self.scroll_view)
        self.layout.add_widget(self.user_name_input)
        self.layout.add_widget(self.user_description_input)
        self.layout.add_widget(self.save_button)
        self.layout.add_widget(self.delete_profile_button)
        self.layout.add_widget(self.chat_button)
        self.add_widget(self.layout)
        
        self.load_profiles()

    def load_profiles(self):
        print("Loading profiles in ProfileSelectionScreen...")
        profile_dir = "E:\\Alita\\Profiles\\User"
        self.grid_layout.clear_widgets()
        for profile_file in os.listdir(profile_dir):
            if profile_file.endswith(".json"):
                try:
                    with open(os.path.join(profile_dir, profile_file), "r") as file:
                        profile_data = json.load(file)
                        btn = Button(text=profile_data["name"], size_hint_y=None, height=44)
                        btn.bind(on_release=lambda btn=btn, profile_data=profile_data: self.select_profile(profile_data))
                        self.grid_layout.add_widget(btn)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {profile_file}")

    def select_profile(self, profile_data):
        print(f"Selecting profile: {profile_data}")
        self.user_name_input.text = profile_data["name"]
        self.user_description_input.text = profile_data["description"]
        self.profile_manager.set_profile(profile_data)

    def save_profile(self, instance):
        user_name = self.user_name_input.text.strip()
        user_description = self.user_description_input.text.strip()
        
        if user_name and user_description:
            profile_data = {
                "name": user_name,
                "description": user_description
            }
            profile_file = os.path.join("E:\\Alita\\Profiles\\User", f"{user_name}.json")
            os.makedirs(os.path.dirname(profile_file), exist_ok=True)
            with open(profile_file, "w") as file:
                json.dump(profile_data, file)
            
            # Reload the profile into memory
            print(f"Saving profile: {profile_data}")
            self.profile_manager.set_profile(profile_data)
            self.load_profiles()  # Reload profiles to reflect changes

    def delete_profile(self, instance):
        profile_name = self.user_name_input.text.strip()
        if profile_name:
            print(f"Deleting profile: {profile_name}")
            self.profile_manager.delete_profile(profile_name)
            self.user_name_input.text = ""
            self.user_description_input.text = ""

    def switch_to_chat(self, instance):
        print("Switching to chat screen...")
        self.manager.current = 'chat'

if __name__ == '__main__':
    print("Starting the application...")
    # Add your application initialization code here
