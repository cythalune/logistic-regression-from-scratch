import numpy as np
import matplotlib.pyplot as plt
from src.model import predict


def plot_data(X, Y, mean, std):
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.figure(figsize=(8, 5), dpi=100)

    # Unstandardize the features for plotting
    X = (X * std) + mean  

    # Separate data points by class
    # 1 = Malignant, 0 = Benign
    X_malignant = X[Y == 1]
    X_benign = X[Y == 0]

    # Plot each class separately to generate distinct legend handles
    plt.scatter(X_malignant[:, 0], X_malignant[:, 1], color='#e41a1c', s=20, alpha=0.6, label='Malignant')
    plt.scatter(X_benign[:, 0], X_benign[:, 1], color='#377eb8', s=20, alpha=0.6, label='Benign')

    plt.xlabel("Radius mean (\u03bcm)", fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel("Texture mean", fontsize=11, fontweight='bold', labelpad=10)
    
    plt.title('Scatter Plot of Data', fontsize=14, fontweight='bold', pad=15)
    
    # Your exact legend styling now works for both classes
    plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10)
    plt.tight_layout()

    plt.savefig('plots/data_plot.png', dpi=300, bbox_inches='tight')
    plt.show()



def plot_cost(cost_history):
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.plot(cost_history, color='#1f77b4', linewidth=2.5)

    plt.ylabel("Cost", fontsize=11, fontweight='bold', labelpad=10)
    plt.xlabel("Number of iterations", fontsize=11, fontweight='bold', labelpad=10)
    plt.title("Cost Function During Training")

    plt.savefig('plots/cost_plot.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(X, Y, w, b, mean, std):
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(8, 5), dpi=100)

    # Convert standardized data back to original values for plotting
    X_real = (X * std) + mean

    # Plot data points
    plt.scatter(
        X_real[:, 0],
        X_real[:, 1],
        c=Y,
        cmap='coolwarm',
        s=20,
        alpha=0.6
    )

    # Create a grid in original feature values
    x_min = X_real[:, 0].min() - 1
    x_max = X_real[:, 0].max() + 1
    y_min = X_real[:, 1].min() - 1
    y_max = X_real[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1)
    )

    # Turn the grid into (number_of_points, 2)
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # Standardize the grid because the model was trained
    # on standardized features
    X_grid_std = (X_grid - mean) / std

    # Predict probability for every point in the grid
    Z = predict(w, b, X_grid_std)

    Z = Z.reshape(xx.shape)

    # Draw the 0.5 probability boundary
    plt.contour(
        xx,
        yy,
        Z,
        levels=[0.5],
        colors='black',
        linestyles='--',
        linewidths=2
    )

    plt.xlabel("Radius mean (μm)", fontsize=11, fontweight='bold')
    plt.ylabel("Texture mean", fontsize=11, fontweight='bold')
    plt.title("Decision Boundary", fontsize=14, fontweight='bold')

    plt.tight_layout()

    plt.savefig(
        'plots/decision_boundary.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()