import json
import os
import random
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

random.seed(42)
np.random.seed(42)

MAPPING_DIR = '../mappings/'

s_values_df = pd.read_csv(MAPPING_DIR+"goodreads_s_values_uniform.csv")

cols_to_keep = ['user_number', 'item_number', 's1', 's2', 's3', 's4']

s_values_df = s_values_df[cols_to_keep]

s_values_df['user_number'] = s_values_df['user_number'].apply(lambda x: str(x))
s_values_df['item_number'] = s_values_df['item_number'].apply(lambda x: str(x))
s_values_df['user_item_id'] = s_values_df['user_number'] + "-" + s_values_df['item_number']


print("Loading Data")

OUTPUT_DATA_DIR = "../output_data/"

train_df_processed = pd.read_csv(OUTPUT_DATA_DIR+"text_processed_training.csv")
val_df_processed = pd.read_csv(OUTPUT_DATA_DIR+"text_processed_validation.csv")
test_df_processed = pd.read_csv(OUTPUT_DATA_DIR+"text_processed_testing.csv")

train_df = pd.concat([train_df_processed, val_df_processed], axis=0)
train_df = train_df.sample(frac=0.1)


def load_mapping(mapping_file):
    """Loads the mapping from `mapping_file`.

    Parameters
    ----------
    mapping_file: str
        The name of the mapping file to import.

    Returns
    -------
    pd.DataFrame
        The DataFrame created from the mapping.

    """
    return pd.read_csv(os.path.join("../mappings", "{}.csv".format(mapping_file)))


user_map = load_mapping("user_map")
book_map = load_mapping("book_map")

book_map['book_id'] = book_map['book_id'].apply(lambda x: str(x))


def create_user_item_id(data_df, u_map, i_map):
    """Creates a user-item ID for the records in `data_df`.

    The user-item ID is created from `u_map` and `i_map`.
    Both mappings, map text user IDs to numeric user IDs
    and these numeric user IDs are combined to form the
    user-item ID.

    Parameters
    ----------
    data_df: pd.DataFrame
        The DataFrame for which the user-item ID is created.
    u_map: pd.DataFrame
        A DataFrame containing a mapping from a text user ID to
        a numeric user ID.
    i_map: pd.DataFrame
        A DataFrame containing a mapping from a text item ID to
        a numeric item ID.

    Returns
    -------
    pd.DataFrame
        The DataFrame obtained from `data_df` after adding a
        user-item ID field based on `u_map` and `i_map`.

    """
    data_df['book_id'] = data_df['book_id'].apply(lambda x: str(x))
    data_df = pd.merge(data_df, u_map, how="left", on=["user_id"])
    data_df = pd.merge(data_df, i_map, how="left", on=["book_id"])
    data_df['user_number'] = data_df['user_number'].apply(lambda x: str(x))
    data_df['book_number'] = data_df['book_number'].apply(lambda x: str(x))
    data_df['user_item_id'] = data_df['user_number'] + "-" + data_df['book_number']
    return data_df.drop(columns=['user_number', 'book_number'])


print("Running mappings")


train_df = create_user_item_id(train_df, user_map, book_map)
test_df = create_user_item_id(test_df_processed, user_map, book_map)

s_values_df.drop(columns=['user_number', 'item_number'], inplace=True)

train_df_s = pd.merge(train_df, s_values_df, how='left', on=['user_item_id'])
test_df_s = pd.merge(test_df, s_values_df, how='left', on=['user_item_id'])

train_df_s.drop(columns=['user_item_id'], inplace=True)
test_df_s.drop(columns=['user_item_id'], inplace=True)



print("Feature Selection")

feature_columns = ['text_reviews_count', 'is_ebook', 'average_rating', 'num_pages',
                   'ratings_count', 'is_translated', 'is_in_series', 'series_length',
                   'is_paperback', 'is_hardcover', 'is_audio', 'from_penguin',
                   'from_harpercollins', 'from_university_press', 'from_vintage',
                   'from_createspace', 'publication_year', 'author_a', 'author_b',
                   'author_c', 'author_d', 'author_e', 'author_f',
                   'book_shelved_count', 'shelved_count', 'read_count', 'rated_count',
                   'recommended_count', 'title_len', 's1', 's2', 's3', 's4']
X_train_reg = train_df_s[feature_columns]
X_test_reg = test_df_s[feature_columns]


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


log_transform_cols = ['text_reviews_count', 'ratings_count', 'shelved_count', 'read_count', 'rated_count', 'recommended_count']
X_train_reg = log_transform_columns(X_train_reg, log_transform_cols)
X_test_reg = log_transform_columns(X_test_reg, log_transform_cols)


def sigmoid(val):
    """Applies the sigmoid function to `val`.

    The sigmoid function has the form
    f(x) = 1 / (1 + exp(-x))

    Parameters
    ----------
    val: float
        The operand to the sigmoid function.

    Returns
    -------
    float
        The result of applying the sigmoid
        function to `val`.

    """
    return 1 / (1 + np.exp(-val))


def scale_s_values(data_df):
    """Applies the sigmoid function to the s values in `data_df`.

    Parameters
    ---------
    data_df: pd.DataFrame
        The DataFrame for which the operation is performed.

    Returns
    -------
    pd.DataFrame
        The DataFrame that results from scaling the s values
        in `data_df`.

    """
    for s_col in ["s1", "s2", "s3", "s4"]:
        data_df[s_col] = data_df[s_col].apply(lambda x: sigmoid(x))
    return data_df


min_max_scaler = MinMaxScaler()

X_train_reg2 = min_max_scaler.fit_transform(scale_s_values(X_train_reg))
X_test_reg2 = min_max_scaler.transform(scale_s_values(X_test_reg))

print("Running Word2Vec")

book_df = train_df_s[['book_id', 'cleaned_text']]
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
    text_vecs = [word for word in str(book_text) if word in w2v.wv.vocab]
    if len(text_vecs) > 0:
        return np.mean(w2v[text_vecs], axis=0)
    return np.zeros(vec_length)


print("Creating book vectors")

train_df_s['book_vector'] = train_df_s['cleaned_text'].apply(lambda x: create_book_vector(x, 200))
test_df_s['book_vector'] = test_df_s['cleaned_text'].apply(lambda x: create_book_vector(x, 200))


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


train_wv = create_book_vec_df(train_df_s['book_vector'], train_df_s.index)
test_wv = create_book_vec_df(test_df_s['book_vector'], test_df_s.index)


X_train_wv_reg = pd.concat([train_wv, pd.DataFrame(np.vstack(X_train_reg2))], axis=1)
X_test_wv_reg = pd.concat([test_wv, pd.DataFrame(np.vstack(X_test_reg2))], axis=1)


print("Fitting Logistic Regression")

tuning_params = [{'C': [0.1, 0.3],
                  'max_iter': [10, 20]}]
reg_clf = GridSearchCV(LogisticRegression(), tuning_params, scoring='roc_auc', cv=10)
reg_clf.fit(X_train_wv_reg, train_df_s['recommended'])


X_train_reg_s = scale_s_values(X_train_reg)
X_test_reg_s = scale_s_values(X_test_reg)

X_train_wv_reg1 = pd.concat([train_wv, X_train_reg_s], axis=1)
X_test_wv_reg1 = pd.concat([test_wv, X_test_reg_s], axis=1)


print("Fitting XGBoost")

tuning_params = [{'n_estimators': [10, 20],
                  'max_depth': [1, 2],
                  'learning_rate': [0.1, 0.3]}]
xgb_clf = GridSearchCV(XGBClassifier(), tuning_params, scoring='roc_auc', cv=10)
xgb_clf.fit(X_train_wv_reg1, train_df_s['recommended'])


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


perform_cross_validation(
    reg_clf, "Logistic Regression", X_train_wv_reg,
    train_df_s['recommended'], X_test_wv_reg, test_df_s['recommended'])
print()
print()
print()
perform_cross_validation(
    xgb_clf, "XGBoost", X_train_wv_reg1,
    train_df_s['recommended'], X_test_wv_reg1, test_df_s['recommended'])
