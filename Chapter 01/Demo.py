class Demo:
    def __init__(self, name):
        self.name = name
        print("Started!")

    def hello(self):
        print("Hey " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")

m = Demo("Alexa")
m.hello()
m.goodbye()
