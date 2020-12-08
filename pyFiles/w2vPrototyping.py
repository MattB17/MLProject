import json
import os
import random
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import scipy.sparse as sp

random.seed(42)
np.random.seed(42)

######################################################
# Load Data
######################################################

print("Loading data")

OUTPUT_DATA_DIR = "./output_data/"

train_df_processed = pd.read_csv(OUTPUT_DATA_DIR+"text_processed_training.csv")
val_df_processed = pd.read_csv(OUTPUT_DATA_DIR+"text_processed_validation.csv")

train_df_processed['cleaned_text'] = train_df_processed['cleaned_text'].apply(lambda x: "" if pd.isnull(x) else x)
val_df_processed['cleaned_text'] = val_df_processed['cleaned_text'].apply(lambda x: "" if pd.isnull(x) else x)

######################################################
# Pre Word2Vec Processing
######################################################

print("W2V Pre Processing")

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

print("Building W2V Model")

book_df = train_df_processed[['book_id', 'cleaned_text']]
book_df = book_df.drop_duplicates(subset=['book_id'])

book_df['cleaned_text'] = book_df['cleaned_text'].apply(lambda x: "" if pd.isnull(x) else x)

w2v = Word2Vec(list(book_df['cleaned_text']), size=200, window=10, min_count=1)

def create_book_vector(book_text, vec_length):
    """Creates a vector for the book given by `book_text`.

    The word vectors for each word in `book_text` are
    averaged to build a vector for the book.

    Parameters
    ----------
    book_text: str
        The book text for which the vector is generated.

    Returns
    -------
    vector
        A vector for the book.

    """
    text_vecs = [word for word in book_text if word in w2v.wv.vocab]
    if len(text_vecs) > 0:
        return np.mean(w2v[text_vecs], axis=0)
    return np.zeros(vec_length)

train_df_processed['book_vector'] = train_df_processed['cleaned_text'].apply(lambda x: create_book_vector(x, 200))
val_df_processed['book_vector'] = val_df_processed['cleaned_text'].apply(lambda x: create_book_vector(x, 200))

def create_book_vec_df(book_vecs, indices):
    """Creates a dataframe from `book_vecs`.

    Each numpy array in `book_vecs` is converted to a
    row in the resulting dataframe.

    Parameters
    ----------
    book_vecs: list
        A list of numpy arrays where each array corresponds
        to the book vector for a book.
    indicies: np.array
        A numpy array of indices for the DataFrame

    Returns
    -------
    pd.DataFrame
        The DataFrame obtained from converting `review_vecs`
        to a dataframe.

    """
    book_vec_df = pd.DataFrame(np.vstack(book_vecs))
    book_vec_df.columns = ["word" + str(col) for col in book_vec_df.columns]
    book_vec_df.index = indices
    return book_vec_df

train_wv = create_book_vec_df(train_df_processed['book_vector'], train_df_processed.index)
val_wv = create_book_vec_df(val_df_processed['book_vector'], val_df_processed.index)

X_train_reg_df = pd.DataFrame(np.vstack(X_train_reg))
X_train_reg_df.index = train_df_processed.index

X_val_reg_df = pd.DataFrame(np.vstack(X_val_reg))
X_val_reg_df.index = val_df_processed.index

X_train_wv_reg = sp.csr_matrix(pd.concat([train_wv, X_train_reg_df], axis=1))
X_val_wv_reg = sp.csr_matrix(pd.concat([val_wv, X_val_reg_df], axis=1))

print("Running XGBoost")

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

xg_cls = XGBClassifier(
    objective='binary:logistic', learning_rate=0.1,
    max_depth=2, n_estimators=2000)

xg_cls.fit(X_train_wv_reg, train_df_processed['recommended'])
train_MSE = roc_auc_score(
    train_df_processed['recommended'], xg_cls.predict(X_train_wv_reg))
val_MSE = roc_auc_score(
    val_df_processed['recommended'], xg_cls.predict(X_val_wv_reg))

print("Training AUC: {}".format(train_MSE))
print("Validation AUC: {}".format(val_MSE))
