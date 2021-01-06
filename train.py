import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs
from model import MovielensModel



def load_data(dataset):

    if dataset == 'movielens':

        ratings = tfds.load('movielens/100k-ratings', split="train")
        movies = tfds.load('movielens/100k-movies', split="train")

        # Select the basic features.
        ratings = ratings.map(lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
            "user_rating": x["user_rating"],
        })
        movies = movies.map(lambda x: x["movie_title"])

    elif dataset == 'kaggle_movies':

        ratings = tf.data.experimental.make_csv_dataset('data/ratings_test.csv', batch_size=1, column_defaults=[tf.int32, tf.string, tf.string, tf.float32, tf.int32])
        movies = tf.data.experimental.make_csv_dataset('data/movies_preprocessed.csv', batch_size=1, column_defaults=[tf.string], select_columns=['id'])

        ratings = ratings.map(lambda x: {
            "movie_title": x["movieId"],
            "user_id": x["userId"],
            "user_rating": x["rating"],
        })
        movies = movies.map(lambda x: x["id"])

    return movies, ratings



def main(dataset):

    movies, ratings = load_data(dataset)

    # Randomly shuffle data and split between train and test.
    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)


    if dataset == 'movielens':

        movie_titles = movies.batch(1_000)
        user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

        unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
        unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    elif dataset == 'kaggle_movies':

        movie_titles = movies.batch(1_000)
        user_ids = ratings.batch(1_000).map(lambda x: x["user_id"])

        print('finding uniques')
        unique_movie_titles = np.asarray(pd.read_csv('data/movies_preprocessed.csv').id.astype(int).unique()).astype('str')
        unique_user_ids = np.asarray(pd.read_csv('data/ratings_test.csv').userId.astype(int).unique()).astype('str')
        # unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
        # unique_user_ids = np.unique(np.concatenate(list(user_ids)))
        print('done finding uniques')


    print('initializing model obect')
    model = MovielensModel(movies, unique_user_ids, unique_movie_titles, rating_weight=1.0, retrieval_weight=1.0)
    print('compiling model')
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

    print('caching train data')
    cached_train = train.shuffle(100_000).batch(8192).cache()
    print('caching test data')
    cached_test = test.batch(4096).cache()

    print('commencing model fitting')
    print('cached_train')
    print(cached_train)
    model.fit(cached_train, epochs=3)
    print('evaluating fitted model on test set')
    metrics = model.evaluate(cached_test, return_dict=True)

    print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
    # print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")

    # model = MovielensModel(movies, unique_user_ids, unique_movie_titles, rating_weight=0.0, retrieval_weight=1.0)
    # model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

    # model.fit(cached_train, epochs=3)
    # metrics = model.evaluate(cached_test, return_dict=True)

    # print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
    # print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")

    # model = MovielensModel(movies, unique_user_ids, unique_movie_titles, rating_weight=1.0, retrieval_weight=1.0)
    # model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

    # model.fit(cached_train, epochs=3)
    # metrics = model.evaluate(cached_test, return_dict=True)

    # print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
    # print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")


if __name__ == '__main__':

    # main(dataset='movielens')
    main(dataset='kaggle_movies')