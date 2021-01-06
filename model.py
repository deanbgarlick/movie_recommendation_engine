import tensorflow as tf
import tensorflow_recommenders as tfrs

from typing import Dict, Text


class MovielensModel(tfrs.models.Model):

    def __init__(self, movies, unique_user_ids, unique_movie_titles, rating_weight: float, retrieval_weight: float) -> None:
        # We take the loss weights in the constructor: this allows us to instantiate
        # several model objects with different loss weights.

        super().__init__()

        embedding_dimension = 32

        # User and movie models.
        self.movie_model: tf.keras.layers.Layer = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None), #input_shape=(None,32)),
        tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])
        self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_user_ids, mask_token=None), #input_shape=(None,32)),
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        print('unique_movie_titles.shape')
        print(unique_movie_titles.shape)
        print('unique_user_ids.shape')
        print(unique_user_ids.shape)

        # A small model to take in user and movie embeddings and predict ratings.
        # We can make this as complicated as we want as long as we output a scalar
        # as our prediction.
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),# input_shape=(None,64)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

        # The tasks.
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(lambda x: self.movie_model(tf.reshape(x, [-1]))),
            )
        )

        # The loss weights.
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        # user_id = tf.reshape(features["user_id"], [-1, 32])
        # print('features["user_id"]')
        # print(user_id)
        #user_id = features["user_id"]

        user_id = tf.reshape(features["user_id"], [-1])
        print('features["user_id"]')
        print(user_id)
        user_embeddings = self.user_model(user_id)

        # And pick out the movie features and pass them into the movie model.
        #movie_title = tf.reshape(features["movie_title"], [-1, 32])
        # print('features["movie_title"]')
        # print(movie_title)
        #movie_title = features["movie_title"]

        movie_title = tf.reshape(features["movie_title"], [-1])
        print('features["movie_title"]')
        print(movie_title)
        movie_embeddings = self.movie_model(movie_title)

        print('user_embeddings')
        print(user_embeddings)
        print('movie_embeddings')
        print(movie_embeddings)

        print('tf.concat([user_embeddings, movie_embeddings], axis=1)')
        print(tf.concat([user_embeddings, movie_embeddings], axis=1))

        # print('tf.concat([user_embeddings, movie_embeddings]')
        # print(tf.concat([user_embeddings, tf.reshape(movie_embeddings, [-1, 32])], axis=1))

        print('feeding into rating model')
        print(
            self.rating_model(
                tf.concat([user_embeddings, movie_embeddings], axis=1)
            )
        )

        return (
            user_embeddings,
            movie_embeddings,
            # We apply the multi-layered rating model to a concatentation of
            # user and movie embeddings.
            self.rating_model(
                tf.concat([user_embeddings, movie_embeddings], axis=1)
            ),
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

        ratings = tf.reshape(features.pop("user_rating"), [-1])

        print('ratings')
        print(ratings)

        user_embeddings, movie_embeddings, rating_predictions = self(features)

        print('now have features')
        print('user_embeddings')
        print(user_embeddings)
        print('movie_embeddings')
        print(movie_embeddings)
        print('rating_predictions')
        print(rating_predictions)

        print('computing rating_loss')
        # We compute the loss for each task.
        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating_predictions,
        )

        print('computing retrieval_loss')
        retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)

        print('returning')
        # And combine them using the loss weights.
        return (self.rating_weight * rating_loss
                + self.retrieval_weight * retrieval_loss)
