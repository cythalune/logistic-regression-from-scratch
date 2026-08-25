# Logistic Regression from Scratch

A simple implementation of logistic regression using gradient descent in Python.

The model is trained on the Breast Cancer Wisconsin dataset using two features: mean radius and mean texture.

## Features

- Logistic regression implemented from scratch
- Sigmoid activation
- Binary cross-entropy cost
- Gradient descent optimization
- Feature standardization
- Train/test split
- Evaluation metrics:
  - Confusion Matrix
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Training loss visualization
- Decision boundary visualization

## Project Structure

```text
.
├── data/
│   └── data.csv
├── plots/
│   ├── cost_plot.png
│   ├── data_plot.png
│   └── decision_boundary.png
├── src/
│   ├── data.py
│   ├── model.py
│   ├── metrics.py
│   └── visualize.py
├── main.py
└── requirements.txt
```

## How to Run

1. Clone this repository and navigate to the project folder.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the Kaggle dataset files are placed in the `data/` directory (see the [Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download) section).
4. Run the main execution script:
   ```bash
   python main.py
   ```

## Purpose
This project was built to understand logistic regression by implementing the main components manually rather than using a machine learning library.