import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def run_tsne(x: pd.DataFrame, y: pd.DataFrame, out_file: str = None):
    # Initialize and fit t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(x)

    # Create a DataFrame for the t-SNE results
    df_tsne = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
    df_tsne["Target"] = y

    # Plot t-SNE results
    plt.figure(figsize=(10, 8))

    # Scatter plot with color map according to 'Target'
    sc = plt.scatter(df_tsne["TSNE1"], df_tsne["TSNE2"], c=df_tsne["Target"], cmap="viridis", s=100, alpha=0.7)

    # Add color bar
    plt.colorbar(sc, label="Target Value")

    # Add titles and labels
    plt.title("t-SNE Visualization with Target Heatmap")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    if out_file:
        plt.savefig(out_file)
    else:
        plt.show()
