import pandas as pd
import numpy as np


class BehavioralUtilities:
    def process_csv(df):
        # df is actually a row being inputed
        def get_row_idx():
            return df.name
            # print(df.name, "type:",type(df.name))

        """ 
            The raw ABET file gives us all possible descriptors of the trial, 
        but has another column that tells us whether this descriptor is actually 
        true for this trial. For these functions, any() goes through the series (determined
        by how you grouped the table) and if it finds a True or non-zero value, it's output
        will be according to whatever task we are tracking for this trial. One series
        is a list of values taken from various rows from the raw data in which that rows' labels
        for trials was the same. For example, if there were 165 trials identified for the session,
        then there will be 165 series' for a given behavior/descriptor. We get one output per each function
        that alots the value to the same row as the trial number that it found it. Every function is replacing
        the column for that series we are adding onto the 165 rows that are waiting to be filled by a series.
        """

        def get_block_num():

            if (df[df["Item_Name"] == "Session1"]["Arg1_Value"]).any():
                return 1
            elif (df[df["Item_Name"] == "Session2"]["Arg1_Value"]).any():
                return 2
            elif (df[df["Item_Name"] == "Session3"]["Arg1_Value"]).any():
                return 3

        def get_force_or_free():
            # print(df["Item_Name"] == "Forced-Choice Trials Begin")
            if (df["Item_Name"] == "Forced-Choice Trials Begin").any():
                return "Forced"
            elif (df["Item_Name"] == "Free-Choice Trials begin").any():
                return "Free"

        def get_rew_size():
            # print (df[df["Item_Name"] == "Feeder #2"]["Arg1_Value"])
            if df[df["Item_Name"] == "Feeder #2"][
                "Arg1_Value"
            ].empty:  # meaning there was no value found for feeder #2 or Arg1_Vlaue in the series. so most likely an omission
                return np.nan
            elif (
                len((df[df["Item_Name"] == "Feeder #2"]["Arg1_Value"])) == 1
            ):  # when the series is not empty and has only one value (arg1_value is the amount given)
                if (
                    float(df[df["Item_Name"] == "Feeder #2"]["Arg1_Value"].values[0])
                    < 1.2
                ):
                    return "Small"
                elif (
                    float(df[df["Item_Name"] == "Feeder #2"]["Arg1_Value"].values[0])
                    >= 1.2
                ):
                    return "Large"
            else:  # if series contains two values, means mouse was fed twice, not good
                print(
                    "Multiple rewards in trial %s: %s"
                    % (
                        df[df["Item_Name"] == "Feeder #2"]["trial_num"].values[0],
                        df[df["Item_Name"] == "Feeder #2"]["Arg1_Value"].values,
                    )
                )
                return np.nan

        def get_iftrial_possible():
            """As indicated by "TTL#\d', is the time at which ABET tells ISX software that there is a trial available.
            There should only be value of 0.001s at the beginning of each session and all later values for TTL should be 1.0s
            (meaning only the first trial should have 2 TTLs show up and later ones should only have 1 TTL show up)
            Caveat: It's common that we get two TTLs for each trial, we are ignoring the 2nd TTL that appears per each trial
            by selecting values[0] in the series of values.
            """

            # print(df[df["Item_Name"].str.match("TTL")]["Arg1_Value"].values)
            # print(len(df[df["Item_Name"].str.contains("TTL")]["Arg1_Value"].values))
            if df[df["Item_Name"].str.contains("TTL")]["Arg1_Value"].empty:
                return np.nan
            elif len(df[df["Item_Name"].str.contains("TTL")]["Arg1_Value"]) == 1:
                return df[df["Item_Name"].str.contains("TTL")]["Arg1_Value"].values[0]
            else:
                """print(
                    "Multiple ABET 'trial possible' signals to ISX in trial %s: %s"
                    % (
                        df[df["Item_Name"].str.contains("TTL")]["trial_num"].values[0],
                        df[df["Item_Name"].str.contains("TTL")]["Arg1_Value"].values,
                    )
                )"""
                return df[df["Item_Name"].str.contains("TTL")]["Arg1_Value"].values[0]

        def get_trial_start_time():

            # print(df.loc[df["Item_Name"].str.contains("Trials Begin", case=False)]) #doesn't account for case when finding this pattern of string
            if df[df["Item_Name"].str.contains("Trials Begin", case=False)].empty:
                return np.nan
            # len(df[df["Item_Name"].str.contains("Trials Begin", case=False)]) == 1
            elif len(df[df["Item_Name"].str.contains("Trials Begin", case=False)]) == 1:
                return df[df["Item_Name"].str.contains("Trials Begin", case=False)][
                    "Evnt_Time"
                ].values[0]
            else:  # if the trial contains mulitple start times
                print(
                    "Multiple trial start times in trial %s: %s"
                    % (
                        df[df["Item_Name"].str.contains("Trials Begin", case=False)][
                            "trial_num"
                        ].values[0],
                        df[df["Item_Name"].str.contains("Trials Begin", case=False)][
                            "Evnt_Time"
                        ].values,
                    )
                )
                return np.nan

        def get_choice_time():
            """Uses the output of get_rew_sizes() because, a reward is mouse-initiated, so there can't be a choice if rew_size is NaN.
            Regardles whter the choice was large or small, just get the event time of when that choice was made.
            """
            if df[df["Item_Name"] == "Feeder #2"][
                "Arg1_Value"
            ].empty:  # meaning there was no value found for feeder #2 or Arg1_Vlaue in the series. so most likely an omission
                return np.nan
            elif (
                len((df[df["Item_Name"] == "Feeder #2"]["Arg1_Value"])) == 1
            ):  # when the series is not empty and has only one value (arg1_value is the amount given)
                return df[df["Item_Name"] == "Feeder #2"]["Evnt_Time"].values[0]
            else:
                print(
                    "Multiple choice times in trial %s : %s"
                    % (
                        df[df["Item_Name"] == "Feeder #2"]["trial_num"].values[0],
                        df[df["Item_Name"] == "Feeder #2"]["Evnt_Time"].values,
                    )
                )
                return df[df["Item_Name"] == "Feeder #2"]["Evnt_Time"].values[0]

        def get_collection_time():
            if df[df["Item_Name"].str.contains("Reward Retrieved")].empty:
                return np.nan
            elif len(df[df["Item_Name"].str.contains("Reward Retrieved")]) == 1:
                return df[df["Item_Name"].str.contains("Reward Retrieved")][
                    "Evnt_Time"
                ].values[0]
            else:
                print(
                    "Multiple rewards retrieved in trial %s: %s"
                    % (
                        df[df["Item_Name"].str.contains("Reward Retrieved")][
                            "trial_num"
                        ].values[0],
                        df[df["Item_Name"].str.contains("Reward Retrieved")][
                            "Evnt_Time"
                        ].values,
                    )
                )
                return df[df["Item_Name"].str.contains("Reward Retrieved")][
                    "Evnt_Time"
                ].values[0]

        """def shocked():

            if df[df["Item_Name"] == "shock_on_off"][
                "Arg1_Value"
            ].empty:  # there wasn't a shock string found, so shock wasnt recorded (shock impossible, so false)
                return False
            elif (
                len(df[df["Item_Name"] == "shock_on_off"]["Arg1_Value"]) == 1
            ):  # only one shock occurred in trial
                shock_value = df[df["Item_Name"] == "shock_on_off"][
                    "Arg1_Value"
                ].values[0]
                if shock_value == 0:
                    return False
                else:
                    return True
            else:
                print(
                    "Multiple shocks in trial %s: %s"
                    % (df[df["Item_Name"] == "shock_on_off"]["trial_num"].values[0]),
                    df[df["Item_Name"] == "shock_on_off"]["Arg1_Value"].values,
                )
                shock_value = df[df["Item_Name"] == "shock_on_off"][
                    "Arg1_Value"
                ].values[0]
                if shock_value == 0 or shock_value == np.nan:
                    return False
                else:
                    return True"""

        def shocked():
            """A shock value of 0.0 means"""

            if any(
                df["Item_Name"].str.contains("Shocker #1")
            ):  # there wasn't a shock string found, so shock wasnt recorded (shock impossible, so false)
                return True
            else:
                return False

        def omission():
            # one will always be empty, so has to be "and"
            # print((df[df["Item_Name"] == "ITI TO (Blank touch or Omission)"]).any())
            result = None
            # the below condition will be true sometimes even though it is ITI omission, so don't return anything yet
            if (
                df[
                    df["Item_Name"].str.contains("Omission of a Free Trial", case=False)
                ].empty
                and df[
                    df["Item_Name"].str.contains(
                        "Omission of a Forced Trial", case=False
                    )
                ].empty
                and (df["Item_Name"] == "ITI TO (Blank touch or Omission)").any()
                == False
            ):
                result = np.nan
            # sometimes ABET never puts down neither "Omission of a Free Trial" nor "Omission of a Forced Trial" and it's still an ITI, so can appear regardless whether those "Omission..." strings appear
            # if this string also exists in the trial
            elif (df["Item_Name"] == "ITI TO (Blank touch or Omission)").any():
                result = "ITI"
            elif (
                len(
                    df[
                        df["Item_Name"].str.contains(
                            "Omission of a Free Trial", case=False
                        )
                    ]
                )
                == 1
                or len(
                    df[
                        df["Item_Name"].str.contains(
                            "Omission of a Forced Trial", case=False
                        )
                    ]
                )
                == 1
            ):
                result = "Omission"
            else:
                result = "Omission"

            return result

        def win_or_loss():
            """If received "Large" reward and received shock as "False", it's a win.
            If received "Large" reward and received shock as "True", it's a loss.
            """
            if force_or_free == "Free":
                if rew_size == "Large" and shocked == False:
                    return "Win"
                elif rew_size == "Large" and shocked == True:
                    return "Loss"
                else:
                    return np.nan

        """All these are just one value, pertaining to one row for this particular trial the apply func is on"""
        block_num = get_block_num()  # list of block numbers
        force_or_free = (
            get_force_or_free()
        )  # list of whether trials were forced or free
        rew_size = (
            get_rew_size()
        )  # list of rew size, number of elements corresponds to number of trials
        trial_possible = (
            get_iftrial_possible()
        )  # is trial possible (bool): indicated by isx telling behavioral software that a trial is possible
        start_time = get_trial_start_time()
        choice_time = get_choice_time()
        collection_time = get_collection_time()
        shocked = shocked()
        omission = omission()  # regardless whether trial was forced or free
        win_loss = win_or_loss()  # list indicating whether the trial was win or loss

        if omission == "Omission":
            # if it's an omission trial, choice time should be +30 of start time
            choice_time = start_time + 30
            print(f"OMISSION TIME: {choice_time}")
        elif omission == "ITI":
            # if it's an ITI trial, find choice time of blankside in raw abet
            # we have a series of values for this trial remember
            if any(df["Item_Name"].str.contains("ITI TO", case=False)):
                time_of_touch = df[df["Item_Name"].str.contains("ITI TO", case=False)][
                    "Evnt_Time"
                ].values[0]
                print(f"TIME OF ITI TOUCH: {time_of_touch}")

                choice_time = time_of_touch

            else:
                print(
                    "NO 'BlankSide' nor 'LeftBlank' nor 'RightBlank' string found for ITI!"
                )

        """This series is added on to the waiting new grouped table in a variable I indicated"""
        return pd.Series(
            {
                "Block": block_num,
                "Trial Type": force_or_free,
                "Reward Size": rew_size,
                "Trial Possible (s)": trial_possible,
                "Start Time (s)": start_time,
                "Choice Time (s)": choice_time,
                "Collection Time (s)": collection_time,
                "Shock Ocurred": shocked,
                "Omission": omission,
                "Win or Loss": win_loss,
            }
        )

    def add_winstay_loseshift_loseomit(df):
        """If previous trial was a win, and current trial they recieved "Large" reward, its a win-stay.
        I previous trial was a loss, and current trial they recieved "Small" reward, its a lose-shift.
        Both of the trials need to be free because if they have no choice, they can't have any stratergy.
        """
        learning_strats = []
        # identify previous trial
        learning_strats.append(np.nan)  # first row always nan for leaning stratergy
        for row_idx in range(1, len(df)):  # skip the first row, always will be nan
            if (
                df.iloc[row_idx - 1][9] == "Win"
                and df.iloc[row_idx][2] == "Large"
                and df.iloc[row_idx][1] == "Free"
            ):  # win stay
                learning_strats.append("Win Stay")
            elif (
                df.iloc[row_idx - 1][9] == "Loss"
                and df.iloc[row_idx][2] == "Small"
                and df.iloc[row_idx][1] == "Free"
            ):  # lose shift
                learning_strats.append("Lose Shift")
            elif (
                df.iloc[row_idx - 1][9] == "Loss"
                and (df.iloc[row_idx][8] == "ITI" or df.iloc[row_idx][8] == "Omission")
                and df.iloc[row_idx][1] == "Free"
            ):  # lose omit, col 8 is whether omission happened right?
                learning_strats.append("Lose Omit")
            else:
                learning_strats.append(np.nan)
        print(learning_strats)
        df["Learning Stratergy"] = learning_strats
        return df

    def shift_col_values(df):
        df["Block"] = df["Block"].shift(1)
        return df

    def interpolate_block(df, trails_in_block):

        count = 1
        curr_block = 1
        # df = df.fillna(value="-") <- not needed
        for row_i in range(1, len(df)):
            if (
                df.loc[row_i, "Omission"] != "ITI"
                and df.loc[row_i, "Omission"] != "Omission"
            ):  # keep counting, only if the omission value is not neither ITI and Omission
                count += 1
            if (
                count == trails_in_block
            ):  # but if that number turns out to 30, change it back to 1 and this means we've moved on to next block
                count = 1
                curr_block += 1
            # the only purpose of count is to change the curr block value
            if np.isnan(df.loc[row_i]["Block"]):
                print("empty row at: ", row_i)
                df.at[row_i, "Block"] = curr_block

        """print("Working on...", df)
        count = 0
        curr_block = 1
        # df = df.fillna(value="-") <- not needed
        for row_i in range(0, len(df)):

            if (
                count == trails_in_block
            ):  # but if that number turns out to 30, change it back to 1 and this means we've moved on to next block
                count = 1
                curr_block += 1
            # can start at count = 0 because it becomes 1 when it sees a number, so there should be 30 of them
            if ((df.loc[row_i, "Omission"] != "ITI") == True) and (
                (df.loc[row_i, "Omission"] != "Omission") == True
            ):  # if both are true in that they are not ITI or Omission, then count
                count += 1
            if not any(
                (
                    count == trails_in_block,
                    (
                        ((df.loc[row_i, "Omission"] != "ITI") == True)
                        and ((df.loc[row_i, "Omission"] != "Omission") == True)
                    ),
                )
            ):
                df.at[row_i, "Block"] = curr_block"""

        return df

    def del_first_row(df):
        df = df[1:]
        return df

    def verify_table(df):
        for row_idx in range(1, len(df)):  # skip trial_num 0
            if (
                df.iloc[row_idx]["Reward Size"] == np.nan
                and df.iloc[row_idx]["Choice Time"] == np.nan
                and df.iloc[row_idx]["Collection Time"] == np.nan
                and df.iloc[row_idx]["Omission"] == np.nan
            ):
                print("Something wrong in row %s" % (row_idx))
        print("All good!")
