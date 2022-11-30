import os

def main():

        list_of_combos_we_care_about = [
            
            "Block_Trial_Type_Start_Time_(s)",
            "Block_Reward_Size_Start_Time_(s)",
            "Block_Reward_Size_Shock_Ocurred_Start_Time_(s)",
            "Block_Trial_Type_Reward_Size_Start_Time_(s)",
            "Block_Trial_Type_Shock_Ocurred_Start_Time_(s)",

        ]

        processed = 0

        ROOT = r"/media/rory/RDT VIDS/BORIS_merge"

        session_paths = []

        for folder in os.listdir(ROOT):
            if "RRD" in folder and "_" not in folder:
                session_paths.append(os.path.join(ROOT, folder))

        ROOT_2 = r"/media/rory/RDT VIDS/BORIS"

        for folder in os.listdir(ROOT_2):
            if "RRD" in folder and "_" not in folder:
                session_paths.append(os.path.join(ROOT_2, folder))
                                
        session_paths = set(session_paths)
        print("num session paths:",len(session_paths))
        print(session_paths)

main()