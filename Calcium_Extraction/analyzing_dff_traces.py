import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def average(lst):
    return sum(lst) / len(lst)


def plot_csv(csv_file_path, out_plot_path):
    means = []
    cell_names = []
    df = pd.read_csv(csv_file_path, sep=",")
    # dropping first row

    df = df.drop([0, 1])
    # print(df.head())

    for col in df:
        cell_names.append(col)
        list1 = []
        for i in df[col]:
            list1.append(float(i))

        mean = average(list1)
        means.append(mean)

    for count, value in enumerate(means):
        print(count, value)
    print(len(means))
    plt.bar(list(range(len(means))), means)
    plt.ylabel("Mean df/f")
    plt.xlabel("Cell #")
    plt.xticks(means)

    plt.savefig(out_plot_path)


def main():
    csv_path = r"/media/rory/RDT VIDS/PTP_Inscopix_#1/BLA-Insc-1/Session-20210216-171723-BLA-Insc-1/dff_traces.csv"
    out_path = r"/media/rory/RDT VIDS/PTP_Inscopix_#1/BLA-Insc-1/Session-20210216-171723-BLA-Insc-1/dff_traces_avg_plot.pdf"
    plot_csv(csv_path, out_path)


main()
