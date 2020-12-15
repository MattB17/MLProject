# Goodreads ChainRec Augmentation

### Authors

* Farid Zandi Shafagh
* Matt Buckley

### Description

This code is in fulfillment of the final course project for CSC2515 at the University of Toronto.

We trained a set of models on the [goodreads dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) to predict whether or not a user would recommend a given book. A recommendation is defined as a rating greater than 3. In particular, the predictive task was as follows: given a user `u` and a book `i`, will user `u` recommend book `i`. Due to resource limitations, we restricted our focus to the poetry category.

In building our models we wanted to combine more traditional recommender system approaches with current state-of-the-art methods. Specifically, we leveraged the [chainRec model](http://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18b.pdf). We first trained the chainRec model on the dataset of interest and modified [the code](https://github.com/MattB17/chainRec) to output a subset of the weights obtained by training the model. These weights were then used as input features to our models. We combined these with user level features, book level features, and text features generated with the Word2Vec model.

The weights captured from chainRec were the `s` variables, as explained in the paper. These represent a ranking of users' preferences for the various books in the dataset. There are four `s` variables for each user-item pair. These represent the users preference to shelve, read, rate, and recommend the given book. So higher `s` values correspond to a higher preference.

### Code Structure

The code is organized into the following sections:

##### 1) `ExplorationAndCleaning`
These notebooks contain the code for initial exploratory analysis and data cleaning.
* `Exploratory.ipynb` contains the initial exploratory analysis
* `DataCleaning.ipynb` and `TextCleaning.ipynb` were used to filter data, preprocess data, and engineer features for model fitting
* `MakeChainRecData.ipynb` contains the code used to build the input dataset used to train the chainRec model

##### 2) `DataSplitting`
These notebooks contain the code to split the data into training, testing, and validation sets. The chainRec source code automatically splits the data into training, testing, and validation. To be consistent with this we allowed chainRec to split the data the first time and then added a method to chainRec in which it loads the data split from a file. In this way we can ensure consistency between our models and the chainRec models as we trained chainRec with various sampling techniques.
* `TrainTestValSplit.ipynb` is used to split the dataset into training, testing, and validation according to the split done by chainRec
* `ChainRecTestValDataset.ipynb` is used to create a csv file in the form chainRec uses to decide the train, test, and validation split

##### 3) `Baselines`
These notebooks contain code for the baseline models.
* `BasicModels.ipynb` contains code for the baseline models of Logistic Regression, Support Vector Classifier, and Random Forest
* `CollaborativeFiltering.ipynb` contains the code for an item-item collaborative filtering model based on kNN

##### 4) `MetaModels`
These notebooks contain the code used to combine the baseline models with the `s` variables trained using chainRec plus the text vectors obtained from `Word2Vec`.
* `MetaChainRecUniform.ipynb` builds these models using the `s` variables from training chainRec using uniform sampling.
* `MetaChainRecStage.ipynb` builds these models using the `s` variables from training chainRec using stagewise sampling.
* `MetaBprMF.ipynb` builds these models using the `s` variables from training bprMF.

##### 5) `FinalModelsEvaluation.ipynb`
In this notebook we run our final candidate models on the combination of testing and validation data. These models are then evaluated on the test set.

##### Other Code
* `pyFiles` contains python files used to run cross validation in order to choose hyperparameters to train final models.
* `mappings` contains a collection of mapping files used to map data between our models and the chainRec models.
  * `book_map.csv` and `user_map.csv` contain mappings from IDs as text to numeric IDs. The dataset contains user IDs and book IDs as strings, but the chainRec source code requires the IDs to be in the form of an increasing sequence starting from 0.
  * `goodreads_train.csv`, `goodreads_val.csv`, and `goodreads_test.csv` contain the numeric user IDs and book IDs of the interactions that appear in the training, validation, and testing datasets, respectively.
  * `goodreads_s_values_uniform.csv`, `goodreads_s_values_stage.csv`, and `goodreads_s_values_bpr.csv` contain the `s` variable values for each user-item interaction in the dataset from training chainRec with uniform sampling, chainRec with stagewise sampling, and bprMF, respectively.
* `Prototyping` contains a few notebooks used to perform some preliminary prototyping. 
