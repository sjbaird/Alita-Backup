from kivy.uix.label import Label
from kivy.core.clipboard import Clipboard

class SelectableLabel(Label):
    def __init__(self, **kwargs):
        super(SelectableLabel, self).__init__(**kwargs)
        self.markup = True
        self.halign = 'left'
        self.valign = 'middle'
        self.bind(size=self.setter('text_size'))
        print("SelectableLabel initialized with markup, halign, and valign set.")

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            Clipboard.copy(self.text)
            print(f"Text copied to clipboard: {self.text}")
        return super(SelectableLabel, self).on_touch_down(touch)

if __name__ == '__main__':
    print("SelectableLabel class defined")
