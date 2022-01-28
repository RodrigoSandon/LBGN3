from Mouse import Mouse


class MouseDatabase:

    mice_dict = dict()

    def __init__(self, database_name):
        self.database_name = database_name
        self.number_of_mice = len(self.mice_list)

    def add_mouse(self, name, mouse: Mouse):

        if name not in self.mice_dict:
            return "Mouse already exists!"
        else:
            self.mice_dict[name] = mouse

    def get_mouse(self, name) -> Mouse:
        return self.mice_dict[name]
