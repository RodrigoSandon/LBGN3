from ISXTable import ISXTable


class Driver:
    def main():
        ex1_dff_path = r"/media/rory/RDT VIDS/PTP_Inscopix_#1/BLA-Insc-3/Session-20210120/dff_traces.csv"
        # make sure to omit for columns starting at 1 (0 index taken up by time so don't omit it)
        dff_1 = ISXTable(ex1_dff_path, [1, 2, 3])
        print(dff_1.processed_isx_csv.head())
        print(dff_1.structure_mouse_name)


if __name__ == "__main__":
    Driver.main()
