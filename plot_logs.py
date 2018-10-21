import matplotlib.pyplot as plt
import sys


def read_logs(f):
    f = open(f, 'r')
    x = []
    y = []

    lines = f.read().splitlines()
    for line in lines:
        fields = line.split(' ')
        id = int(fields[0])
        value = float(fields[1])
        x.append(id)
        y.append(value)

    f.close()
    return x, y


def plot_logs(loss_file, avg_loss_file, score_file, val_loss_file):

    X = []
    Y = []
    labels = ['loss', 'avg_loss', 'score', 'val_loss']
    colors = ['b', 'g', 'r', 'c']
    # marker_sizes = [1, 1, 3, 3]

    x, y = read_logs(loss_file)
    X.append(x)
    Y.append(y)

    x, y = read_logs(avg_loss_file)
    X.append(x)
    Y.append(y)

    x, y = read_logs(score_file)
    X.append(x)
    Y.append(y)

    x, y = read_logs(val_loss_file)
    X.append(x)
    Y.append(y)

    plt.figure()
    plt.title('Training Status')
    plt.ylim((0, 1))
    plt.xlabel('iteration')

    for i, (x, y) in enumerate(zip(X, Y)):
        plt.plot(x, y, color=colors[i], label=labels[i])

    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_logs(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
