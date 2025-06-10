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

In order to train a model, the **training2.py** file is setup to be run as a script. Hyperparameters such as:
- pretrained BERT model
- evaluation type
- learning rate
- weight decay
- number of epoches
- freezing BERT layers except classification head
- etc

can be set using command line arguments. Files **main_confidence.py** and **tuning2.py** can also be run as scripts, however, the later does not take in command line arguments (as it does a hyperparameter sweep). 