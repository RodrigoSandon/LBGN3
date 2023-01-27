import datajoint as dj
dj.config['database.host'] = 'localhost'
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'twiceblackpink9.'

schema = dj.schema('RDBMS', locals())

@schema
class Mouse(dj.Manual):
    definition = """
        # mouse
        mouse_id: char(11)           #unique mouse_id
        ---

    """

mouse = Mouse() 

@schema
class Session(dj.Manual):
    definition = """
    # experiment session
    -> Mouse
    session_date: date            # session date
    ---
    experiment_id: char(20)         # experiment id
    """

session = Session() 