import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default='TestData_Path.csv')


if __name__ == "__main__":
    args = parser.parse_args()
    mae = []
    with open(args.path) as f:
        reader = csv.reader(f)
        for r in reader:
            mae.append(float(r[1]))

    trial_num = np.arange(1, len(mae) + 1)
    plt.plot(trial_num, mae)
    plt.savefig("lr_curv.png", dpi=150)
