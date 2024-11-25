class VariableWatcher:
    def __init__(self):
        self._value = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value
        self.on_change(new_value)

    def on_change(self, new_value):
        print(f"Variable value changed to: {new_value}")