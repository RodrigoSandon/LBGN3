from Session import Session


class Mouse:

    sessions = dict()

    def __init__(self, name):

        self.name = name
        self.number_of_sessions = 0

    def get_mouse_name(self):
        return self.name()

    def add_session(self, id, session: Session):
        if id in self.sessions:
            return "Session already exists!"
        else:
            self.sessions[id] = session

    def get_session(self, id) -> Session:
        return self.sessions[id]
