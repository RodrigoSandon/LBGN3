"""print("Way 2")
        x = list(df[cell_x])
        y = list(df[cell_y])

        start_1 = time.time()
        cost = dtw(x, y, global_constraint="sakoe_chiba", sakoe_chiba_radius=3)
        print("Alignment cost: {:.4f}".format(cost))

        end_1 = time.time()
        print(f"Time taken: {end_1 - start_1}")

        print("Way 3")

        start_2 = time.time()
        cost = dtw(x, y, global_constraint="itakura", itakura_max_slope=2.)
        print("Alignment cost: {:.4f}".format(cost))

        end_2 = time.time()
        print(f"Time taken: {end_2 - start_2}")"""

from pathlib import Path

x = Path(
    "/media/rory/Padlock_DT/BLA_Analysis/PTP_Inscopix_#4/BLA-Insc-9/RDT D1/SingleCellAlignmentData/C02/Block_Reward Size_Choice Time (s)/(2.0, 'Large')/plot_ready.csv"
)

print(x.parts)
"""print(type(x.parts))
print(list(x.parts)[11])"""


def strip_outcome(my_str):
    outcome_bins: list
    if "(" in my_str:
        my_str = my_str.replace("(", "")
    if ")" in my_str:
        my_str = my_str.replace(")", "")
    if "'" in my_str:
        my_str = my_str.replace("'", "")
    if ", " in my_str:
        outcome_bins = my_str.split(", ")

    print(outcome_bins)
    return outcome_bins


print(type(list(x.parts)[11]))

strip_outcome(list(x.parts)[11])
