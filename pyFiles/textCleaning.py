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

train_df = pd.read_csv(OUTPUT_DATA_DIR+"interactions_training.csv")

val_df = pd.read_csv(OUTPUT_DATA_DIR+"interactions_validation.csv")

test_df = pd.read_csv(OUTPUT_DATA_DIR+"interactions_testing.csv")


######################################################
# Text Cleaning Functions
######################################################

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def process_book_text(book_text, exclude_text, ps):
    """Pre-processes the text given by `review_text`.

    Parameters
    ----------
    book_text: str
        The book text to be processed.
    exclude_text: collection
        A collection of words to be excluded.
    ps: PorterStemmer
        The PorterStemmer used to perform word stemming.

    Returns
    -------
    str
        A string representing the processed version of `review_text`.

    """
    book = re.sub('[^a-zA-Z0-9]', ' ', book_text).lower().split()
    book = [ps.stem(word) for word in book if not word in exclude_text]
    return ' '.join(book)


def preprocess_for_classification(data_df):
    """Preprocesses `data_df` to be used in classification.

    Parameters
    ----------
    data_df: pd.DataFrame
        The DataFrame to be processed.

    Returns
    -------
    pd.DataFrame
        The DataFrame obtained from `data_df` after processing.

    """
    # flags for most popular formats
    data_df['format'] = data_df['format'].apply(lambda x: str(x).lower())
    data_df['is_paperback'] = data_df['format'].apply(lambda x: int("paper" in x))
    data_df['is_hardcover'] = data_df['format'].apply(lambda x: int("hard" in x))
    data_df['is_audio'] = data_df['format'].apply(lambda x: int("audio" in x))
    data_df['is_other_format'] = (data_df['is_paperback'] + data_df['is_hardcover'] +
                                  data_df['is_audio'] + data_df['is_ebook'])
    data_df['is_other_format'] = data_df['is_other_format'].apply(lambda x: 0 if x > 0 else 1)

    #flags for most popular publishers
    data_df['publisher'] = data_df['publisher'].apply(lambda x: str(x).lower())
    data_df['from_penguin'] = data_df['publisher'].apply(lambda x: int("penguin" in x))
    data_df['from_harpercollins'] = data_df['publisher'].apply(lambda x: int("harpercollins" in x or "harper collins" in x))
    data_df['from_university_press'] = data_df['publisher'].apply(lambda x: int("university press" in x))
    data_df['from_vintage'] = data_df['publisher'].apply(lambda x: int("vintage" in x))
    data_df['from_createspace'] = data_df['publisher'].apply(lambda x: int("createspace" in x or "create space" in x))
    data_df['other_publisher'] = (data_df['from_penguin'] + data_df['from_harpercollins'] +
                                  data_df['from_university_press'] + data_df['from_vintage'] + data_df['from_createspace'])
    data_df['other_publisher'] = data_df['other_publisher'].apply(lambda x: 0 if x > 0 else 1)

    # ensuring columns are not missing
    train_df['average_rating'] = train_df['average_rating'].apply(lambda x: 0.0 if pd.isnull(x) else x)
    train_df['text_reviews_count'] = train_df['text_reviews_count'].apply(lambda x: 0 if pd.isnull(x) else x)
    train_df['ratings_count'] = train_df['ratings_count'].apply(lambda x: 0 if pd.isnull(x) else x)
    median_page_count = train_df['num_pages'].median()

    train_df['num_pages'] = train_df['num_pages'].apply(lambda x: median_page_count if pd.isnull(x) else x)

    # flags for most popular authors
    train_df['main_author'] = train_df['main_author'].astype(str)
    train_df['author_a'] = train_df['main_author'].apply(lambda x: int(x == "435477.0"))
    train_df['author_b'] = train_df['main_author'].apply(lambda x: int(x == "903.0"))
    train_df['author_c'] = train_df['main_author'].apply(lambda x: int(x == "947.0"))
    train_df['author_d'] = train_df['main_author'].apply(lambda x: int(x == "4624490.0"))
    train_df['author_e'] = train_df['main_author'].apply(lambda x: int(x == "18540.0"))
    train_df['author_f'] = train_df['main_author'].apply(lambda x: int(x == "8075577.0"))
    train_df['author_other'] = (train_df['author_a'] + train_df['author_b'] +
                                train_df['author_c'] + train_df['author_d'] +
                                train_df['author_e'] +train_df['author_f'])
    train_df['author_other'] = train_df['author_other'].apply(lambda x: 0 if x > 0 else 1)
    return train_df


def preprocess_all_book_text(data_df, id_col, text_col, exclude_text, ps):
    """Preprocesses the book text in `data_df` for `text_col`.

    The dataframe is restricted to `id_col` and `text_col` and then the
    unique ids are chosen. This is so that we only preprocess the text
    for a book once. Then we join the resulting text back to `data_df`.

    Parameters
    ----------
    data_df: pd.DataFrame
        The DataFrame containing the data to be preprocessed.
    id_col: str
        The column from which unique ids are chosen.
    text_col: str
        The column to be pre-processed.
    exclude_text: collection
        A collection of words to remove
    ps: PorterStemmer
        The PorterStemmer used for word stemming.

    Returns
    -------
    pd.DataFrame
        The DataFrame obtained from `data_df` after adding a column
        with the processed text.

    """
    book_df = data_df[[id_col, text_col]]
    book_df = book_df.drop_duplicates(subset=[id_col])
    book_df['cleaned_text'] = book_df[text_col].apply(lambda x: process_book_text(x, exclude_text, ps))
    final_df = pd.merge(data_df, book_df[[id_col, "cleaned_text"]], how="inner", on=[id_col])
    return final_df


def run_preprocess_pipeline(data_df, exclude_text, ps):
    """Runs the full pre-processing pipeline on `data_df`.

    Parameters
    ----------
    data_df: pd.DataFrame

    """
    processed_df = preprocess_for_classification(data_df)
    return preprocess_all_book_text(processed_df, "book_id", "title_description", exclude_text, ps)


######################################################
# Running Text Cleaning Pipeline
######################################################

exclude_english = set(stopwords.words('english'))
ps = PorterStemmer()
train_df_processed = run_preprocess_pipeline(train_df, exclude_english, ps)
val_df_processed = run_preprocess_pipeline(val_df, exclude_english, ps)
test_df_processed = run_preprocess_pipeline(test_df, exclude_english, ps)


######################################################
# Saving Data
######################################################

from sklearn.utils import shuffle

def shuffle_dataset(data_df):
    """Randomly shuffles `df`.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to be shuffled.

    Returns
    -------
    pd.DataFrame
        A shuffled dataframe obtained from `df`.

    """
    data_df = shuffle(data_df)
    data_df.reset_index(inplace=True, drop=True)
    return data_df


train_df_processed = shuffle_dataset(train_df_processed)
val_df_processed = shuffle_dataset(val_df_processed)
test_df_processed = shuffle_dataset(test_df_processed)

train_df_processed.to_csv(OUTPUT_DATA_DIR+"text_processed_training.csv")
val_df_processed.to_csv(OUTPUT_DATA_DIR+"text_processed_validation.csv")
test_df_processed.to_csv(OUTPUT_DATA_DIR+"text_processed_testing.csv")
