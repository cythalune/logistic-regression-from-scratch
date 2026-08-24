import numpy as np
import matplotlib.pyplot as plt
from src.model import predict

def plot_data(X, Y):
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.figure(figsize=(8, 5), dpi=100)
    
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='coolwarm', s=20, alpha=0.6, label='Data Points')
    
    plt.ylabel("y", fontsize=11, fontweight='bold', labelpad=10)
    plt.xlabel("x", fontsize=11, fontweight='bold', labelpad=10)
    plt.title('Scatter Plot of Data', fontsize=14, fontweight='bold', pad=15)
    
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

def plot_decision_boundary(X, Y, w, b):
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(8, 5), dpi=100)

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='coolwarm', s=20, alpha=0.6)

    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                          np.arange(y_min, y_max, 0.1))
    Z = predict(np.c_[xx.ravel(), yy.ravel()], w, b)
    Z = Z[:, 1]  
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)

    plt.ylabel("x2", fontsize=11, fontweight='bold', labelpad=10)
    plt.xlabel("x1", fontsize=11, fontweight='bold', labelpad=10)
    plt.title("Decision Boundary", fontsize=14, fontweight='bold', pad=15)

    plt.savefig('plots/decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

     