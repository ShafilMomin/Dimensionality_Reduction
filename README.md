# Dimensionality Reduction on Wine Dataset

A comparison of three classical dimensionality reduction techniques — **PCA**, **LDA**, and **Kernel PCA** — applied to the Wine dataset. Each technique compresses 13 chemical features down to 2 components, which are then fed into a Logistic Regression classifier for customer segment prediction.

---

## Techniques Covered

| Notebook | Method | Type |
|---|---|---|
| `principal_component_analysis.ipynb` | Principal Component Analysis (PCA) | Unsupervised, linear |
| `linear_discriminant_analysis.ipynb` | Linear Discriminant Analysis (LDA) | Supervised, linear |
| `kernel_pca.ipynb` | Kernel PCA (RBF kernel) | Unsupervised, non-linear |

---

## Dataset — `Wine.csv`

178 wine samples with 13 chemical measurements, classified into 3 customer segments.

| Feature | Description |
|---|---|
| `Alcohol` | Alcohol content |
| `Malic_Acid` | Malic acid content |
| `Ash` | Ash content |
| `Ash_Alcanity` | Alkalinity of ash |
| `Magnesium` | Magnesium content |
| `Total_Phenols` | Total phenols |
| `Flavanoids` | Flavanoid content |
| `Nonflavanoid_Phenols` | Non-flavanoid phenols |
| `Proanthocyanins` | Proanthocyanins |
| `Color_Intensity` | Color intensity |
| `Hue` | Hue |
| `OD280` | OD280/OD315 of diluted wines |
| `Proline` | Proline |
| `Customer_Segment` | **Target** — class 1, 2, or 3 |

---

## Pipeline (same across all three notebooks)

```
Load Wine.csv
    ↓
Train/Test Split  (80% / 20%, random_state=0)
    ↓
Feature Scaling   (StandardScaler)
    ↓
Dimensionality Reduction  (→ 2 components)
    ↓
Logistic Regression
    ↓
Confusion Matrix + Accuracy Score
    ↓
Decision Boundary Visualisation
```

---

## Key Differences Between Methods

**PCA** finds directions of maximum variance in the feature space, ignoring class labels entirely. Good general-purpose baseline.

**LDA** finds directions that best *separate* the classes — it uses the target labels during fit, making it a supervised reduction. Often yields cleaner class boundaries than PCA when classes are linearly separable.

**Kernel PCA** applies the kernel trick (here, RBF) to map data into a higher-dimensional space before performing PCA. Useful when class boundaries are non-linear.

---

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```

Python 3.7+ recommended.

---

## Usage

Clone the repo and place `Wine.csv` in the same directory as the notebooks, then run any notebook cell-by-cell:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
jupyter notebook
```

Open whichever notebook you want (`principal_component_analysis.ipynb`, `linear_discriminant_analysis.ipynb`, or `kernel_pca.ipynb`) and run all cells.

---

## File Structure

```
.
├── Wine.csv
├── principal_component_analysis.ipynb
├── linear_discriminant_analysis.ipynb
└── kernel_pca.ipynb
```
