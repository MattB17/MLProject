import json
import os
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

######################################################
# Load Data
######################################################

OUTPUT_DATA_DIR = "./output_data/"

train_df_processed = pd.read_csv(OUTPUT_DATA_DIR+"text_processed_training.csv").sample(frac=0.25)
val_df_processed = pd.read_csv(OUTPUT_DATA_DIR+"text_processed_validation.csv")

######################################################
# Pre-TF-IDF Processing
######################################################

columns_to_keep = ['text_reviews_count', 'is_ebook', 'average_rating', 'num_pages',
                   'publication_year', 'ratings_count', 'is_translated', 'is_in_series',
                   'series_length', 'is_paperback', 'is_hardcover', 'is_audio', 'is_other_format',
                   'from_penguin', 'from_harpercollins', 'from_university_press', 'from_vintage',
                   'from_createspace', 'other_publisher', 'author_a', 'author_b', 'author_c',
                   'author_d', 'author_e', 'author_f', 'author_other']
X_train_reg = train_df_processed[columns_to_keep]
X_val_reg = val_df_processed[columns_to_keep]


def log_transform_columns(data_df, cols):
    """Applies a log transform to `cols` in `data_df`.

    Parameters
    ----------
    data_df: pd.DataFrame
        The DataFrame in which the columns will be transformed.
    cols: collection
        The columns in `data_df` to be log scaled.

    Returns
    -------
    pd.DataFrame
        The DataFrame obtained from `data_df` after log scaling
        the columns `cols`.

    """
    for col in cols:
        data_df[col] = data_df[col].apply(lambda x: np.log(x) if x > 0 else 0)
    return data_df


log_transform_cols = ['text_reviews_count', 'ratings_count']
X_train_reg = log_transform_columns(X_train_reg, log_transform_cols)
X_val_reg = log_transform_columns(X_val_reg, log_transform_cols)

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()

X_train_reg = min_max_scaler.fit_transform(X_train_reg)
X_val_reg = min_max_scaler.transform(X_val_reg)


######################################################
# Prototyping the Textual Model
######################################################

book_df = train_df_processed[['book_id', 'cleaned_text']]
book_df = book_df.drop_duplicates(subset=['book_id'])

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_model = TfidfVectorizer()

tfidf_model.fit(book_df['cleaned_text'])

train_tfidf = tfidf_model.transform(train_df_processed['cleaned_text'])
val_tfidf = tfidf_model.transform(val_df_processed['cleaned_text'])

import scipy.sparse as sp

X_train_reg_sp = sp.csr_matrix(X_train_reg)
X_train_tfidf_reg = sp.hstack((train_tfidf, X_train_reg_sp), format='csr')

X_val_reg_sp = sp.csr_matrix(X_val_reg)
X_val_tfidf_reg = sp.hstack((val_tfidf, X_val_reg_sp), format='csr')

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

learning_rates = [0.03, 0.05, 0.1, 0.3, 0.5]
estimators = [50, 100, 200, 500, 1000, 2000]
depths = [1, 2, 5]

lrs = []
estim = []
ds = []
train_MSEs = []
val_MSEs = []

for learning_rate in learning_rates:
    for estimator in estimators:
        for depth in depths:
            lrs.append(learning_rate)
            estim.append(estimator)
            ds.append(depth)
            print("Learning Rate: {0}, # Estimators: {1}, Depth: {2}".format(learning_rate, estimator, depth))
            print("---------------------------------------------------")
            xg_cls = XGBClassifier(
                objective='binary:logistic', learning_rate=learning_rate,
                max_depth=depth, n_estimators=estimator)
            xg_cls.fit(X_train_tfidf_reg, train_df_processed['recommended'])
            train_MSEs.append(roc_auc_score(
                train_df_processed['recommended'], xg_cls.predict(X_train_tfidf_reg)))
            val_MSEs.append(roc_auc_score(
                val_df_processed['recommended'], xg_cls.predict(X_val_tfidf_reg)))
            print("Training AUC: {}".format(train_MSEs[-1]))
            print("Validation AUC: {}".format(val_MSEs[-1]))
            print()


RESULTS_DIR = './results/'

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


xgb_tfidf = pd.DataFrame({'learning_rate': lrs,
                          'n_estimators': estim,
                          'max_depth': ds,
                          'training_MSE': train_MSEs,
                          'validation_MSE': val_MSEs})
xgb_tfidf.to_csv(RESULTS_DIR+"xgbTFIDF.csv", index=False)
