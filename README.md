# Atlas Frequency ML Assignment

These tasks were presented in the **Introduction to Machine Learning** course as Assignment 2, to practice and get familiar with classical ML, CNNs, and ANN-based anomaly detection.

The assignment consists of three independent machine learning tasks:

---

# Task Descriptions

## Task 1: Image Classification — The Eye Deceives

The first task is an image classification problem.

The dataset contains labelled **32 × 32 RGB images**. Each image belongs to one of two classes, encoded as `0` and `1`.

The goal is to build a PyTorch model that correctly classifies unseen images.

### What you need to do

For this task, I built a CNN model for binary image classification.

The solution includes:

* image preprocessing
* normalization to `[0, 1]`
* PyTorch `Dataset` and `DataLoader`
* CNN architecture
* batch normalization
* dropout regularization
* train/validation split
* validation accuracy calculation
* model saving with `submission_helper`

### Data

The data for this task is stored in:

```text
data/train_data.npz
data/test_data_public.npz
```

The training archive contains:

```text
X: image tensor with shape (10000, 32, 32, 3)
y: labels with shape (10000,)
```

The model expects input in the following format:

```text
(N, 3, 32, 32)
```

where pixel values are normalized to `[0, 1]`.

### Model

Main notebook:

```text
notebooks/task1_cnn_image_classification.ipynb
```

Saved model:

```text
models/task1_model.pkl
```

---

## Task 2: High-Dimensional Tabular Classification — The Petrova Line

The second task is a binary classification problem with many numerical features.

Each row represents one observation, and the input contains **2000 numerical features**.

The goal is to classify unseen observations using a scikit-learn compatible model.

### What you need to do

For this task, I built a tabular ML pipeline for high-dimensional binary classification.

The solution includes:

* CSV data loading with Pandas
* feature selection
* Random Forest classifier
* scikit-learn `Pipeline`
* feature importance analysis
* model saving with `submission_helper`

### Data

The data for this task is stored in:

```text
data/task2_train.csv
data/task2_test_public.csv
```

The training file contains:

```text
X_0 ... X_1999
y
```

where `y` is the binary target label.

### Model

Main notebook:

```text
notebooks/task2_tabular_classification.ipynb
```

Saved model:

```text
models/task2_model.pkl
```

---

## Task 3: ANN Anomaly Detection — The Mechanic's Budget

The third task is an anomaly detection problem using sensor readings.

Each observation contains **20 sensor features**, and the target label indicates whether the observation is normal or anomalous.

The goal is to build a compact neural network anomaly detector using PyTorch.

### What you need to do

For this task, I built a fully connected ANN model for binary anomaly detection.

The solution includes:

* data preprocessing
* train/validation split
* compact feed-forward neural network
* batch normalization
* dropout regularization
* `BCEWithLogitsLoss`
* threshold tuning
* F1-score validation
* model saving with `submission_helper`

### Data

The data for this task is stored in:

```text
data/task3_train.csv
data/task3_test_public.csv
```

The training file contains:

```text
sensor_0 ... sensor_19
anomaly
```

where:

```text
0 = normal
1 = anomaly
```

### Model

Main notebook:

```text
notebooks/task3_ann_anomaly_detection.ipynb
```

Saved model:

```text
models/task3_model.pkl
```

---

# Repository Structure

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

# Technologies Used

* Python
* PyTorch
* Torchvision
* Scikit-learn
* Pandas
* NumPy

---

# Learning Objectives

This assignment helped practice:

* image classification with CNNs
* tabular classification with classical ML models
* ANN-based anomaly detection
* PyTorch model design
* data preprocessing
* train/validation evaluation
* model serialization for automated grading
