# Atlas Frequency ML Assignment

Machine learning assignment project covering image classification, tabular classification, and compact neural anomaly detection.

The repository contains solutions for three independent ML tasks implemented using PyTorch and Scikit-learn.

---

## Tasks

### Task 1 — CNN Image Classification

Binary image classification problem using 32×32 RGB images.

Implemented:

* convolutional neural network (CNN)
* batch normalization
* dropout regularization
* data augmentation
* train/validation evaluation

Notebook:

```text
notebooks/task1_cnn_image_classification.ipynb
```

---

### Task 2 — Tabular Classification

High-dimensional binary classification task.

Implemented:

* Random Forest feature selection
* Random Forest classifier
* scikit-learn pipeline
* feature importance analysis

Notebook:

```text
notebooks/task2_tabular_classification.ipynb
```

---

### Task 3 — ANN Anomaly Detection

Compact feed-forward neural network for anomaly detection.

Implemented:

* fully connected ANN
* batch normalization
* dropout
* BCEWithLogitsLoss
* threshold tuning
* F1-score evaluation

Notebook:

```text
notebooks/task3_ann_anomaly_detection.ipynb
```

---

## Repository Structure

```text
Atlas-Frequency-ML/
├── data/
│   ├── task2_test_public.csv
│   ├── task2_train.csv
│   ├── task3_test_public.csv
│   ├── task3_train.csv
│   ├── test_data_public.npz
│   └── train_data.npz
│
├── models/
│   ├── task1_model.pkl
│   ├── task2_model.pkl
│   └── task3_model.pkl
│
├── notebooks/
│   ├── task1_cnn_image_classification.ipynb
│   ├── task2_tabular_classification.ipynb
│   └── task3_ann_anomaly_detection.ipynb
│
├── docs/
│   └── assignment.pdf
│
├── submission_helper.py
├── .gitignore
└── README.md
```

---

## Technologies

* Python
* PyTorch
* Torchvision
* Scikit-learn
* NumPy
* Pandas
