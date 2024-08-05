import matplotlib.pyplot as plt
import pandas as pd


def analyze_features(x, y):
    df = pd.concat([x, y], axis=1)

    # Plot histograms for each feature
    df.hist(bins=10, figsize=(12, 8), grid=False)

    # Add title and labels
    plt.suptitle("Histograms of Each Feature")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Show the plot
    plt.show()
