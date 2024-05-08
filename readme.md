# Decision Tree (DT) Classifier

This repository contains code for training a DT classifier using a toy dataset.

## Overview

The `dtm_run.py` script loads a toy dataset from a CSV file, preprocesses the data, splits it into training and testing sets, trains a DT classifier, and evaluates its performance. Additionally, it saves the trained model, accuracy, confusion matrix, and decision tree visualization to separate files.

## Contents

- `toys.csv`: CSV file dataset.
- `dtm_run.py`: Python script for training and evaluating the DT classifier.
- `./train/X_train.csv`, `./test/X_test.csv`, `./train/y_train.csv`, `./test/y_test.csv`: CSV files containing the split training and testing datasets.
- `./model/decision_tree_model.pkl`: Saved DT model.
- `./results/results.txt`: Text file containing accuracy and confusion matrix.
- `./views/decision_tree.png`: PNG file containing the decision tree visualization.

## Usage

1. Install the required packages listed in `requirements.txt` using the command `pip install -r requirements.txt`.
2. Run the `dtm_run.py` script to train the decision tree classifier and evaluate its performance.
3. Check the generated files for the trained model, accuracy, confusion matrix, and decision tree visualization.

## Requirements

The code requires the following Python packages:

contourpy==1.2.1
cycler==0.12.1
fonttools==4.51.0
importlib_resources==6.4.0
joblib==1.4.2
kiwisolver==1.4.5
matplotlib==3.8.4
numpy==1.26.4
packaging==24.0
pandas==2.2.2
pillow==10.3.0
pyparsing==3.1.2
python-dateutil==2.9.0.post0
pytz==2024.1
scikit-learn==1.4.2
scipy==1.13.0
six==1.16.0
threadpoolctl==3.5.0
tzdata==2024.1
zipp==3.18.1


You can install all of these packages using pip or package managers of your choice.


## Author

Kushal Pokhrel

## License

This project is licensed under the MIT License - visit to know more on the MIT LICENSE website for details.
