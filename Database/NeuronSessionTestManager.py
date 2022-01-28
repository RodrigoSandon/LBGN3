"""
    Purpose:
        To be able to serialize and deserialize this neuron objects as an input (blob)/output
        (python object) for writing/quereying single cell information via SQLite/Python.

        This way, I can run different analyses (at different times) while still keeping a record
        of the information gained from previous analyses.

    Attributes:
        1) Cell name (includes mice name + local cell #)
        2) A dictionary:
            [session+analysis+event_type] : [identity (+,-, or NEUTRAL)]
                                          .
                                          .
                                          .                              
        
    """


class NeuronSessionTestManager:
    def __init__(self, name):
        self.name = name
        self.mouse_origin = name.split("_")[0]
        self.local_cell_num = name.split("_")[1]
        self.identities = {}

    def add_id(
        self,
        analysis: str,
        event_type: str,
        sample_size: int,
        subwindow_base: str,
        subwindow_post: str,
        standardize: bool,
        smooth: bool,
        identity: str,
    ):
        key_name = "_".join(
            analysis,
            event_type,
            sample_size,
            subwindow_base,
            subwindow_post,
            standardize,
            smooth,
        )

        if key_name in self.identities:
            print("Error: key already exists in identities dictionary!")
        else:
            self.identities[key_name] = identity

        return key_name
