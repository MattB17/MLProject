# Goodreads ChainRec Augmentation

### Authors

* Farid Zandi Shafagh
* Matt Buckley

### Description

This code is in fulfillment of the final course project for CSC2515 at the University of Toronto.

We trained a set of models on the [goodreads dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) to predict whether or not a user would recommend a given book. A recommendation is defined as a rating greater than 3. In particular, the predictive task was as follows: given a user `u` and a book `i`, will user `u` recommend book `i`. Due to resource limitations, we restricted our focus to the poetry category.

In building our models we wanted to combine more traditional recommender system approaches with current state-of-the-art methods. Specifically, we leveraged the [chainRec model](http://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18b.pdf). We first trained the chainRec model on the dataset of interest and modified [the code](https://github.com/MattB17/chainRec) to output a subset of the weights obtained by training the model. These weights were then used as input features to our models. We combined these with user level features, book level features, and text features generated with the Word2Vec model.

The weights captured from chainRec were the `s` variables, as explained in the paper. These represent a ranking of users' preferences for the various books in the dataset. There are four `s` variables for each user-item pair. These represent the users preference to shelve, read, rate, and recommend the given book. So higher `s` values correspond to a higher preference. 
