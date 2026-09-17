# PCA on the Wine Dataset

Reduce **13 numerical features to two principal components**, then use Logistic Regression to predict the dataset's class label.

**Python · pandas · scikit-learn · Matplotlib**

## What is included

- [principal_component_analysis.ipynb](principal_component_analysis.ipynb): the PCA and classification workflow.
- [Wine.csv](Wine.csv): **178 samples**, 13 numerical measurements and `Customer_Segment` labels 1, 2 and 3.

This repository currently contains **PCA only**. LDA and Kernel PCA notebooks are not included. The target column's name is `Customer_Segment`; the model works with the supplied wine-sample classes, not observed shopping behaviour.

## How it works

1. Load the CSV and separate the 13 features from the target.
2. Split into **142 training rows and 36 test rows**, using `random_state=0`.
3. Fit StandardScaler on training data and transform the test data.
4. Fit `PCA(n_components=2)` on training data and transform the test data.
5. Train Logistic Regression on the two components.
6. Print a confusion matrix and accuracy; plot decision boundaries.

PCA creates new combinations of the original features. It keeps directions with high variance and does not use class labels while fitting. It does not simply choose two original columns.

## Verified result

The existing notebook was executed during the September 2026 review. It achieved **97.22% accuracy (35 / 36 correct)** on this test split:

```text
[[14,  0, 0],
 [ 1, 15, 0],
 [ 0,  0, 6]]
```

This matches the saved result. It is one small split, not a claim of general accuracy or proof that PCA improves classification: no full-feature baseline comparison is included. Explained-variance ratios are not reported in the notebook.

## Run it

Open the notebook in Jupyter or Google Colab. Keep its CSV/TSV in the notebook's working folder; opening a notebook from GitHub in Colab does not automatically upload the data.

For a local setup, clone this repository, create and activate a virtual environment, then run:

```bash
python -m pip install numpy pandas matplotlib scikit-learn notebook
python -m notebook
```

Run cells from top to bottom.


The review used Python 3.10, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2 and Matplotlib 3.10.9. Only this README was updated; the notebook and CSV were preserved.
