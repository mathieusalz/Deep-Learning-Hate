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
- **dataset_preprocessing.py**: preprocesses all the data to harmonize categories and saves the train and test datasets to /processed_data
- **data_utils.py**: preprocesses the data for use by the BERT model
- **training.py**: basic training loop for the BERT models
- **training_utils.py**: training utility functions 
- **finetuning.py**: cross-validation hyperparameter tuning of the models
- **main_confidence.py**: runs the model with different seeds to determine its confidence intervals
- **baseline.py**: trains baseline models (logistic regression, decision tree, Linear SVM, Multinomial NB)
- **extBERT.py**: extends the classification head of BERT

The folders are organised as follows:
- **data**: contains the original data
- **processed_data**: contains the data after being processes by **dataset_preprocessing.py**
- **notebooks**: contains notebooks for data analysis of the datasets
- **old_ver**: contains slightly different versions of the python files

# Running the code

In order to train a model, both **training.py** and **main_confidence.py** files are setup to be run as a script. They accept the following configurable arguments:

- **--eval_type**
Specifies the evaluation strategy. Default is "per-lang".

- **--pretrain**
Name or path of the pretrained model to use. Default is "bert-base-multilingual-cased", but other models such as "bert-base-uncased"
or "bert-large-uncased" can be used. The corresponding dataset is automatically loaded.

- **--batch_size**
Batch size used for training and evaluation. Default is 16.

- **--learning_rate**
Learning rate for the optimizer. Default is 4e-6.

- **--num_epochs**
Number of epochs to train the model. Default is 6.

- **--weight_decay**
Weight decay used in AdamW optimizer. Default is 0.1.

- **--freeze**
If set to True, some BERT layers are frozen and not updated during training. Default is False.

- **--debug**
If passed, uses 1% of the dataset.

- **--smallData**
If passed, uses 40% of the dataset.

- **--classImbal**
If passed, enables class imbalance handling strategies. Default is True.

- **--langImbal**
If passed, enables language imbalance handling. Default is True.

The file **tuning.py** can also be run as script, however, it does not take in command line arguments (as it does a hyperparameter sweep). 