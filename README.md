# AI MotionSense — Human Activity Recognition

Human Activity Recognition (HAR) using smartphone sensor data and recurrent neural networks.

This project explores how sequential sensor data can be used to recognize human activities using **SimpleRNN, LSTM, GRU, and Bidirectional LSTM (BiLSTM)** models.

---

## 📌 Project Overview

Smartphones contain sensors such as accelerometers and gyroscopes that continuously capture motion information.

In this project, sensor measurements are treated as **time sequences** rather than independent tabular observations. Recurrent neural networks are then used to learn temporal patterns in these sequences and classify the corresponding human activity.

The project uses the **UCI Human Activity Recognition Using Smartphones Dataset**.

### Activities

The model classifies six activities:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

---

## 🎯 Objectives

- Understand the difference between tabular and sequential data.
- Convert smartphone sensor measurements into sequences suitable for RNN-based models.
- Understand how recurrent neural networks process temporal information.
- Implement and compare:
  - SimpleRNN
  - LSTM
  - GRU
  - Bidirectional LSTM
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
- Analyze where the final model performs well and where classification errors occur.

---

## 📊 Dataset

The project uses the **UCI Human Activity Recognition Using Smartphones Dataset**.

Dataset source:

https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

The original dataset contains smartphone sensor measurements collected from subjects performing six different activities.

For this project, six sensor signals were selected:

- Body Accelerometer X
- Body Accelerometer Y
- Body Accelerometer Z
- Body Gyroscope X
- Body Gyroscope Y
- Body Gyroscope Z

---

## 🔄 Data Representation

Instead of treating each observation as an independent row, the sensor measurements were represented as sequences.

Each sequence contains:

```text
128 timesteps × 6 features
Therefore, the final model input has the shape:

(samples, 128, 6)

Where:

128 = number of timesteps in each sequence
6 = sensor features at each timestep

This allows recurrent neural networks to learn patterns across time.
🧹 Data Preparation

The preprocessing workflow included:

Loading the UCI HAR dataset.
Selecting six sensor signals.
Constructing 128-timestep sequences.
Encoding the activity labels.
Splitting the training data into training and validation sets.
Keeping the official test set separate.
Standardizing the sensor features using the training data.

The test data was kept separate from model training and preprocessing fitting to avoid data leakage.

Dataset Shapes

Original dataset:

Training data: (7352, 561)
Test data:     (2947, 561)

Final sequence representation:

Training set:   (4704, 128, 6)
Validation set: (1177, 128, 6)
Test set:       (2947, 128, 6)
🧠 Model Architectures

Four recurrent architectures were implemented and evaluated.

1. SimpleRNN

A basic recurrent neural network was used as the baseline model.

It processes the sequence step by step and maintains a hidden state containing information from previous timesteps.

2. LSTM

Long Short-Term Memory networks were implemented to better capture dependencies across the sequence.

LSTM uses gating mechanisms to control what information should be retained, updated, or forgotten.

3. GRU

Gated Recurrent Units were implemented as another gated recurrent architecture.

GRU provides a simpler recurrent structure while still allowing the model to capture longer-term dependencies.

4. Bidirectional LSTM

A Bidirectional LSTM was also evaluated.

Instead of processing the sequence in only one direction, the model processes the sequence in both forward and backward directions before producing the final representation.

📈 Model Comparison

The models were evaluated on the untouched test set.

Model	Test Accuracy
SimpleRNN	49.92%
LSTM	63.32%
GRU	62.95%
BiLSTM	70.89%

The BiLSTM achieved the highest test accuracy among the architectures evaluated in this project.

🏆 Final Model

The final model selected for this project is the Bidirectional LSTM (BiLSTM).

Final Test Performance
Test Accuracy: 70.89%
Test Loss:     0.6136
Classification Report
Activity	Precision	Recall	F1-score
Walking	0.98	0.92	0.95
Walking Upstairs	0.95	0.94	0.95
Walking Downstairs	0.90	0.99	0.94
Sitting	0.58	0.23	0.33
Standing	0.58	0.38	0.46
Laying	0.45	0.85	0.59

The model performs particularly well on the three walking activities.

The stationary activities — Sitting, Standing, and Laying — are more difficult for the model to distinguish.

🔍 Confusion Matrix

The confusion matrix provides a detailed view of the final BiLSTM predictions.

Rows represent the actual activity, while columns represent the predicted activity.

The model correctly identifies most samples belonging to the three walking activities.

The larger classification errors occur among the stationary activities:

Sitting
Standing
Laying

For example, a significant number of Sitting and Standing samples were predicted as Laying.

This shows that the main challenge for the final model is distinguishing between similar stationary postures.

📊 Final Confusion Matrix
[[455  15  25   0   0   1]
 [  7 442  20   0   0   2]
 [  1   4 415   0   0   0]
 [  0   3   0 114 105 269]
 [  1   0   0  47 204 280]
 [  0   0   0  37  41 459]]

Class order:

0 → WALKING
1 → WALKING_UPSTAIRS
2 → WALKING_DOWNSTAIRS
3 → SITTING
4 → STANDING
5 → LAYING
🧪 Experimental Results

The progression across the recurrent architectures was:

SimpleRNN
    ↓
49.92% Test Accuracy

LSTM
    ↓
63.32% Test Accuracy

GRU
    ↓
62.95% Test Accuracy

BiLSTM
    ↓
70.89% Test Accuracy

The experiments demonstrate how different recurrent architectures behave when applied to the same sequential sensor representation.

🛠️ Technologies Used
Python
NumPy
Pandas
Scikit-learn
TensorFlow
Keras
Matplotlib
Seaborn
Google Colab
GitHub
📁 Repository Structure
AI-MotionSense-HAR/
│
├── AI_MotionSense_HAR.ipynb
│
├── models/
│   └── bilstm_har_model.keras
│
├── results/
│   ├── bilstm_confusion_matrix.png
│   └── model_comparison.png
│
└── README.md
Files

AI_MotionSense_HAR.ipynb

Contains the complete project workflow, including:

Data loading
Data preprocessing
Sequence construction
Standardization
RNN implementation
LSTM implementation
GRU implementation
BiLSTM implementation
Model comparison
Final evaluation
Confusion matrix
Classification report

models/bilstm_har_model.keras

Saved final Bidirectional LSTM model.

results/model_comparison.png

Visual comparison of test accuracy across the four recurrent architectures.

results/bilstm_confusion_matrix.png

Confusion matrix for the final BiLSTM model.

💡 Key Learning Outcomes

Through this project, I learned how to:

Work with smartphone sensor data.
Understand the difference between tabular and sequential data.
Represent sensor measurements as sequences.
Understand timesteps and sequential features.
Prepare sequence data for recurrent neural networks.
Implement SimpleRNN, LSTM, GRU, and BiLSTM models.
Compare recurrent architectures experimentally.
Evaluate classification models using multiple metrics.
Analyze model errors using confusion matrices.
Avoid test-data leakage during preprocessing.
Save and organize trained deep learning models.
Build an end-to-end deep learning project.
🚀 Future Improvements

Possible future extensions include:

Hyperparameter tuning.
Improving classification of stationary activities.
Exploring additional sensor features.
Experimenting with different sequence lengths.
Testing additional recurrent architectures.
Real-time activity prediction using smartphone sensor streams.
Deploying the trained model for real-world inference.
📚 Dataset Reference

Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L.

Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine.

International Workshop of Ambient Assisted Living (IWAAL 2012).

Dataset:

https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

👩‍💻 Author

Juhi Mishra

BTech — Artificial Intelligence & Data Science


### One small correction before you commit

Your README says the **final model is BiLSTM**, which matches your actual experiment:

**70.89% test accuracy.**

Don't upload the raw UCI dataset to GitHub. Your notebook can document the dataset source, while the repository stays lightweight and clean.

For the GitHub repository, the final structure should be:

```text
AI-MotionSense-HAR/
├── AI_MotionSense_HAR.ipynb
├── models/
│   └── bilstm_har_model.keras
├── results/
│   ├── bilstm_confusion_matrix.png
│   └── model_comparison.png
└── README.md
