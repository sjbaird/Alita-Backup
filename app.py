from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock

from chat_manager import ChatManager
from machine_learning import MLManager  # Use MLManager instead of model_manager
from profile_selection_screen import ProfileManager, ProfileSelectionScreen  # Combined import
from character_selection_screen import CharacterSelectionScreen

class ChatBotApp(App):
    def build(self):
        self.title = "AI Chatbot Interface"

        self.screen_manager = ScreenManager()

        self.chat_screen = Screen(name='chat')

        try:
            print("Initializing ChatManager...")
            chat_manager = ChatManager(screen_manager=self.screen_manager)  # Pass screen_manager as argument
            print("ChatManager initialized.")

            print("Building chat interface...")
            chat_interface = chat_manager.build_chat_interface()
            self.chat_screen.add_widget(chat_interface)
            print("Chat interface built.")

            print("Initializing CharacterSelectionScreen...")
            self.character_selection_screen = CharacterSelectionScreen(name='character_selection', chat_manager=chat_manager)
            print("CharacterSelectionScreen initialized.")

            print("Initializing ProfileManager...")
            chat_manager.profile_manager = ProfileManager(chat_manager.profile_dropdown, self.screen_manager, chat_manager.profile_select_btn)
            chat_manager.profile_manager.load_profiles()
            chat_manager.profile_manager.load_last_profile()  # Ensure ProfileManager loads the last selected profile
            chat_manager.character_manager.load_last_character()
            self.profile_selection_screen = ProfileSelectionScreen(name='profile_selection', profile_manager=chat_manager.profile_manager)
            print("ProfileManager initialized.")

            print("Adding widgets to ScreenManager...")
            self.screen_manager.add_widget(self.chat_screen)
            self.screen_manager.add_widget(self.character_selection_screen)
            self.screen_manager.add_widget(self.profile_selection_screen)
            print("Widgets added to ScreenManager.")

            # Set the initial screen to the chat screen
            self.screen_manager.current = 'chat'
        except Exception as e:
            import traceback
            print(f"An error occurred: {e}")
            traceback.print_exc()
            return None

        return self.screen_manager

def start_app():
    print("Starting ChatBotApp...")
    ChatBotApp().run()
    print("ChatBotApp finished running.")

if __name__ == '__main__':
    print("Initializing models...")
    try:
        ml_manager = MLManager(callback=lambda: Clock.schedule_once(lambda dt: start_app(), 0))
        ml_manager.check_for_new_or_changed_models()
        print("Models initialized.")
    except Exception as e:
        import traceback
        print(f"An error occurred during model initialization: {e}")
        traceback.print_exc()
    print("End of script.")
