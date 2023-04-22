import datajoint as dj
import datetime

# make sure this is in the bla_alignment environment

dj.config['database.host'] = 'localhost'
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'twiceblackpink9.'

schema = dj.schema('RDBMS', locals())

@schema
class Mouse(dj.Manual):
    definition = """
        # mouse
        mouse_id: char(11)           #unique mouse id
        ---

    """

@schema
class Session(dj.Manual):
    definition = """
    # experiment session
    -> Mouse
    session_id = default: char(10)            # session id
    ---
    experiment_name: char(20)         # experiment name
    """

mouse = Mouse() 
session = Session() 

"""class MMDDYYYY(dj.DataJointType):
    def validate(self, value):
        try:
            # Parse date string using strptime
            dt = datetime.datetime.strptime(value, '%m%d%Y').date()
            return dt
        except ValueError:
            # Return None if parsing fails
            return None

    def __str__(self):
        return 'date' """
