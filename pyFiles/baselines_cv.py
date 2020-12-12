import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import auc, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import os
import json


def construct_data_path(dataset_name):
    """Constructs the path to `dataset_name`.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset.

    Returns
    -------
    str
        A path to the dataset.

    """
    return os.path.join('../output_data', '{}.csv'.format(dataset_name))


print("Loading Data")

train_data = pd.read_csv(construct_data_path('interactions_training'))
validation_data = pd.read_csv(construct_data_path('interactions_validation'))
test_data = pd.read_csv(construct_data_path('interactions_testing'))

print("Data Loaded")

train_df = pd.concat([train_data, validation_data], axis=0)


def median_impute(data_df, col_lst):
    """For each column in `col_lst`, missing values are imputed with the median.

    Parameters
    ----------
    data_df: pd.DataFrame
        The DataFrame on which the imputation is done.
    col_lst: collection
        The collection of columns being imputed.

    Returns
    -------
    pd.DataFrame
        The DataFrame obtained from `data_df` after performing the imputation.

    """
    for col in col_lst:
        median_val = data_df[col].median()
        data_df[col] = data_df[col].apply(lambda x: median_val if pd.isnull(x) else x)
    return data_df


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

    data_df = median_impute(data_df, ['num_pages', 'publication_year'])

    # flags for most popular authors
    data_df['main_author'] = data_df['main_author'].astype(str)
    data_df['author_a'] = data_df['main_author'].apply(lambda x: int(x == "435477.0"))
    data_df['author_b'] = data_df['main_author'].apply(lambda x: int(x == "903.0"))
    data_df['author_c'] = data_df['main_author'].apply(lambda x: int(x == "947.0"))
    data_df['author_d'] = data_df['main_author'].apply(lambda x: int(x == "4624490.0"))
    data_df['author_e'] = data_df['main_author'].apply(lambda x: int(x == "18540.0"))
    data_df['author_f'] = data_df['main_author'].apply(lambda x: int(x == "8075577.0"))
    data_df['author_other'] = (data_df['author_a'] + data_df['author_b'] +
                               data_df['author_c'] + data_df['author_d'] +
                               data_df['author_e'] + data_df['author_f'])
    data_df['author_other'] = data_df['author_other'].apply(lambda x: 0 if x > 0 else 1)
    return data_df


def get_total_shelved_count(shelves_str):
    """Find the total number of shelves that a book is in.

    Parameters
    ----------
    str: shelves_str
        a json encoded dictionary contrining the shelving information.

    Returns
    -------
    int
        total number of shelves a book is in.

    """

    total = 0
    shelves_str = shelves_str.replace("\'", "\"")
    shelves_list = json.loads(shelves_str)
    for shelve in shelves_list:
        total += int(shelve['count'])
    return total


def get_title_len(row):
    """Get the length of the title of the book.

    Parameters
    ----------
    DataFrame Row: row
        A row of DataFrame corresponding the the book

    Returns
    -------
    int
        The length of the title. Returns 0 on failure

    """

    try:
        return len(row['title'])
    except:
        return 0


def engineer_features(data):
    """Removes the extra features and adds useful ones.

    Parameters
    ----------
    DataFrame: data
        A DataFrame

    Returns
    -------
    DataFrame:
        A modified DataFrame with one-hot encoded format and publishers,
        total number of shelved instances, length of the title of the book,
        and 0-filled NaN values in numeric fields.
        Also without extra columns and without non-numeric fields.

    """
    data = preprocess_for_classification(data)
    data['book_shelved_count'] = data.apply(lambda row: get_total_shelved_count(row['popular_shelves']), axis=1)
    data['title_len'] = data.apply(lambda row: get_title_len(row), axis=1)

    fillna_columns = ['text_reviews_count','average_rating', 'ratings_count']
    data[fillna_columns] = data[fillna_columns].fillna(0)

    return data

print("Cleaning Data")

train_df_cls = engineer_features(train_df)
test_df_cls = engineer_features(test_data)

print("Data Cleaned")

columns_to_keep = ['text_reviews_count', 'is_ebook', 'average_rating', 'num_pages',
                   'ratings_count', 'is_translated', 'is_in_series', 'series_length',
                   'is_paperback', 'is_hardcover', 'is_audio', 'from_penguin',
                   'from_harpercollins', 'from_university_press', 'from_vintage',
                   'from_createspace', 'author_a', 'author_b', 'author_c',
                   'author_d', 'author_e', 'author_f', 'book_shelved_count', 'title_len',
                   'shelved_count', 'read_count', 'rated_count', 'recommended_count']

y_train = train_df_cls['recommended']
y_test = test_df_cls['recommended']

X_train = train_df_cls[columns_to_keep]
X_test = test_df_cls[columns_to_keep]



print("Fitting Logistic Regression")

tuning_params = [{'C': [0.1, 0.3, 0.5, 1.0, 2.0],
                  'max_iter': [100, 200, 500, 1000, 10000]}]
reg_clf = GridSearchCV(LogisticRegression(), tuning_params, scoring='roc_auc', cv=10)
reg_clf.fit(X_train, y_train)

print("Fitting Random Forest")

tuning_params = [{'n_estimators': [100, 200, 1000, 5000],
                  'max_depth': [1, 5, 10, 15, 20]}]
rf_clf = GridSearchCV(RandomForestClassifier(), tuning_params, scoring='roc_auc', cv=10)
rf_clf.fit(X_train, y_train)

def perform_cross_validation(clf, model_name, X_train_df, y_train_df, X_test_df, y_test_df):
    train_preds = clf.predict(X_train_df)
    test_preds = clf.predict(X_test_df)

    print("{} Parameters".format(model_name))
    print(clf.best_params_)
    print()
    print("Training")
    print("--------")
    print(classification_report(y_train_df, train_preds))
    print("AUC: {}".format(roc_auc_score(y_train_df, train_preds)))
    print()

    print("Testing")
    print("--------")
    print(classification_report(y_test_df, test_preds))
    print("AUC: {}".format(roc_auc_score(y_test_df, test_preds)))


perform_cross_validation(reg_clf, "Logistic Regression", X_train, y_train, X_test, y_test)
print()
print()
print()
perform_cross_validation(rf_clf, "Random Forest", X_train, y_train, X_test, y_test)
