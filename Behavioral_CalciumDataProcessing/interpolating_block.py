import pandas as pd
import numpy as np


def interpolate_block(df, trails_in_block):
    print("Working on...", df)
    count = 0
    curr_block = 1
    # df = df.fillna(value="-") <- not needed
    df = pd.read_csv(df)
    for row_i in range(0, len(df)):
        print("In row:", row_i)
        print("curr_block: ", curr_block)
        print("trail %s in curr block %s" % (count, curr_block))
        print("ITI", df.loc[row_i, "Omission"] == "ITI")
        print("Omission", df.loc[row_i, "Omission"] == "Omission")

        if (
            count == trails_in_block
        ):  # but if that number turns out to 30, change it back to 1 and this means we've moved on to next block
            count = 1
            curr_block += 1
        else:  # can start at count = 0 because it becomes 1 when it sees a number, so there should be 30 of them
            if ((df.loc[row_i, "Omission"] != "ITI") == True) and (
                (df.loc[row_i, "Omission"] != "Omission") == True
            ):  # if both are true in that they are not ITI or Omission, then count
                count += 1
            else:
                df.at[row_i, "Block"] = curr_block

    new_path = abet_path.replace(".csv", "_test.csv")
    df.to_csv(
        new_path, index=True,
    )


abet_path = "/media/rory/PTP Inscopix 2/PTP_Inscopix_#3/BLA-Insc-5/Session-20210610-090307_BLA-Insc-5_RDT_D2_NEW_SCOPE/2021-06-10-09-11-11_video_BLA-Insc-5_RDT_D2_NEW_SCOPE/BLA-INSC-5 06102021_ABET_GPIO_processed.csv"
interpolate_block(abet_path, 30)
