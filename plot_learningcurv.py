import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--title', default='CrowdFlow 0.01')
parser.add_argument('-tmae', '--train_mae', default='TrainMAE.csv')
parser.add_argument('-tloss', '--train_loss', default='Trainloss.csv')
parser.add_argument('-vmae', '--val_mae', default='ValMAE.csv')
parser.add_argument('-vloss', '--val_loss', default='Valloss.csv')


if __name__ == "__main__":
    args = parser.parse_args()
    train_mae = []
    train_loss = []
    val_mae = []
    val_loss = []

    with open(args.train_mae) as f:
        reader = csv.reader(f)
        for r in reader:
            train_mae.append(float(r[1]))

    with open(args.train_loss) as f:
        reader = csv.reader(f)
        for r in reader:
            train_loss.append(float(r[1]))

    with open(args.val_mae) as f:
        reader = csv.reader(f)
        for r in reader:
            val_mae.append(float(r[1]))

    with open(args.val_loss) as f:
        reader = csv.reader(f)
        for r in reader:
            val_loss.append(float(r[1]))

    epochs = np.arange(1, len(train_mae) + 1)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(epochs, train_mae, label="Train")
    ax1.plot(epochs, val_mae, label="Val")
    ax1.set_title('MAE')
    ax1.legend()

    ax2.plot(epochs, train_loss, label="Train")
    ax2.plot(epochs, val_loss, label="Val")
    ax2.set_title('Loss')
    ax2.legend()

    fig.suptitle(args.title)
    fig.savefig("lr_curv.png", dpi=150)
