import matplotlib.pyplot as plt
import pandas as pd


def plot_dataset(name):
    df = pd.read_csv(f"examples/datasets/{name}.csv")
    # Group 0 (blue)
    plt.scatter(
        df[df["label"] == 0]["x"],
        df[df["label"] == 0]["y"],
        color="blue",
        label="Label 0"
    )

    # Group 1 (blue)
    plt.scatter(
        df[df["label"] == 1]["x"],
        df[df["label"] == 1]["y"],
        color="red",
        label="Label 1"
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(name)
    plt.grid(True)

    plt.show()

plot_dataset("circle_regions")