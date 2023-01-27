import datajoint as dj
import pandas as pd

dj.config['database.host'] = 'localhost'
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'twiceblackpink9.'

schema = dj.schema('tutorial', locals())

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
    session_id: char(30)            # session id
    ---
    """

session = Session() 

@schema
class Neuron(dj.Imported):
    definition = """
    -> Session
    cell_id: char(5)       # local cell number
    ---
    activity:  longblob    # dff activity of the neuron
    """

    def make(self, key):
       # use key dictionary to determine the data file path
       session_id = "{session_id}".format(**key)
       session_id_day = session_id.split("_")[0]
       print(session_id_day)
       data_file_1 = "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#1/{mouse_id}/".format(**key)
       data_file_2 = f"{session_id_day}/dff_traces_preprocessed.csv"
       data_file = "".join([data_file_1, data_file_2])

       # load the data
       dff_traces = pd.read_csv(data_file)

       neuron = dff_traces[self.cell_id]

       # add the loaded data as the "activity" column
       key['activity'] = neuron

       # insert the key into self
       self.insert1(key)

       print('Populated a neuron for {mouse_id} on {session_id}'.format(**key))

neuron = Neuron() 
neuron.populate()
