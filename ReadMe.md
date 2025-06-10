# Setup Instructions

## Creating a Virtual Environment with the Required Packages

To create the virtual environment with the proper packages:

1. **Create the virtual environment**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment**
 
   ```bash
    source venv/bin/activate
    ````

3. **Install packages**

    ```bash
    pip install -r requirements.txt
    ```

# Code Structure

This project is made up of the follow python files:
- **dataset_preprocessing.py**: preprocesses all the data to harmonize categories and creates the train and test datasets
- **data_utils2.py**: preprocesses the data for use by the BERT model
- **training2.py**: basic training loop for the BERT models
- **finetuning.py**: cross-validation hyperparameter tuning of the models
- **main_confidence.py**: runs the model with different seeds to determine its confidence intervals
- **baseline.py**: trains baseline models (logistic regression, decision tree, Linear SVM, Multinomial NB)
- **extBERT.py**: extends the classification head of BERT

The folders are organised as follows:
- **data**: contains the original data
- **processed_data**: contains the data after being processes by **dataset_preprocessing.py**
- **notebooks**: contains notebooks for data analysis of the datasets
- **logs**: contains the training run histories
- **results**: contains the console output from training runs
- **old_ver**: contains slightly different versions of the python files