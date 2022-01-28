import json
import pandas as pd

print("BAD JSON")
with open(
    "/media/rory/PTP Inscopix 1/Inscopix/Raw Inscopix Data Files/BLA-Insc-2/Session-20210126/session.json"
) as f:
    data1 = json.load(f)

# print(json.dumps(data1, indent=4, sort_keys=True))
##df1.to_csv("/home/rory/repos/isx/docs/Session-20210126_json.csv", index=False)

print("GOOD JSON")
with open(
    "/media/rory/PTP Inscopix 1/Inscopix/Raw Inscopix Data Files/BLA-Insc-2/Good Sessions/Session-20210217-171510-BLA-Insc-2/session.json"
) as f2:
    data2 = json.load(f2)
print(json.dumps(data2, indent=4, sort_keys=True))

df2 = pd.DataFrame(data2)
df2.to_csv(
    "/media/rory/PTP Inscopix 1/Inscopix/Raw Inscopix Data Files/BLA-Insc-2/Good Sessions/Session-20210217-171510-BLA-Insc-2/session.json",
    index=False,
)
