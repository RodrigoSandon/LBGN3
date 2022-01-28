from Mouse import Mouse
from ISXTable import ISXTable
from MouseDatabase import MouseDatabase
import Utilities


class Driver:
    def main():

        # Create a new mouse database -> -> add processed dff table
        new_db = MouseDatabase("Example Dataset")
        mouse_1_name = "BLA-Insc-1"
        new_db.add_mouse(mouse_1_name, Mouse(mouse_1_name))
        mouse_1 = new_db.get_mouse(mouse_1_name)

        session_path = r"/media/rory/RDT VIDS/PTP_Inscopix_#1/BLA-Insc-1/Session-20210114-171459-BLA-Insc-1"
        session_id = Utilities.parse_session_id(session_path)
        print("Session ID: ", session_id)
        mouse_1.add_session(session_id)
        session_1 = mouse_1.get_session(session_id)

        (
            dff_csv_path,
            motioncorr_isxd_path,
            cellset_isxd_path,
        ) = Utilities.find_add_isx_session_parameters(session_path)
        session_1.add_isx_session(dff_csv_path, motioncorr_isxd_path, cellset_isxd_path)

        # Create a new Mouse object
        mouse_1 = Mouse("BLA-Insc-1")
        dff_traces_path = r"/media/rory/RDT VIDS/PTP_Inscopix_#1/BLA-Insc-1/Session-20210114-171459-BLA-Insc-1/dff_traces.csv"

        # make sure to omit for columns starting at 1 (0 index taken up by time so don't omit it)
        dff_1 = ISXTable(ex1_dff_path, [1, 2, 3])
        print(dff_1.processed_isx_csv.head())
        print(dff_1.structure_mouse_name)

        """
        #/home/rory/Rodrigo/BehavioralDataProcessing/Pho_Vid_Package/74 12042019.csv
        #/home/rory/Rodrigo/BehavioralDataProcessing/BLA-INSC-6 05182021.csv
        ABET_1 = BehavioralSession("BLA-INSC-6 05182021", "/home/rory/Rodrigo/BehavioralDataProcessing/BLA-INSC-6 05182021.csv")
        ABET_1.preprocess_csv()
        df = ABET_1.get_df()
        grouped_by_trialnum = df.groupby("trial_num")
        processed_behavioral_df = grouped_by_trialnum.apply(BehavioralUtilities.process_csv) #is a new df, it's not the modified df
        BehavioralUtilities.add_winstay_loseshift_loseomit(processed_behavioral_df)
        verify = BehavioralUtilities.verify_table(processed_behavioral_df)
        print(verify)
        processed_behavioral_df
        """


if __name__ == "__main__":
    Driver.main()
