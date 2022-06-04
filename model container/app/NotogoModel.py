import tensorflow_recommenders as tfrs
import numpy as np
import tensorflow as tf
from typing import Dict, Text
from Deploy_Retrieval_V2_2 import preprocess_dataset, generate_model_all, inference

class NoToGoModel(tfrs.models.Model):

  def __init__(self, 
                rating_weight: float, 
                like_weight: float,
                retrieval_weight: float
                ) -> None:

    # We take the loss weights in the constructor: this allows us to instantiate
    # several model objects with different loss weights.
    self.BATCH_SIZE = 180
    self.EMB_DIM = 41

    cached_train, cached_test, unique_location_name, unique_user_ids, locations = preprocess_dataset()

    super().__init__()

    embedding_dimension = int(self.EMB_DIM)

    # User and movie models.
    self.location_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_location_name, mask_token=None),
      tf.keras.layers.Embedding(len(unique_location_name) + 1, embedding_dimension),
      tf.keras.layers.Dense(16, activation="relu")
    ])

    self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension),
      tf.keras.layers.Dense(16, activation="relu")
    ])

    # A small model to take in user and movie embeddings and predict ratings.
    # We can make this as complicated as we want as long as we output a scalar
    # as our prediction.
    self.rating_model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation = "sigmoid"),
    ])

    self.like_model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation = "sigmoid"),
    ])

    # The tasks.
    self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    self.like_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=locations.batch(self.BATCH_SIZE).map(self.location_model)
        )
    )

    # The loss weights.
    self.rating_weight = rating_weight
    self.retrieval_weight = retrieval_weight
    self.like_weight = like_weight

  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the movie features and pass them into the movie model.
    location_embeddings = self.location_model(features["location_name"])
    
    return (
        user_embeddings,
        location_embeddings,
        # We apply the multi-layered rating model to a concatentation of
        # user and movie embeddings.
        self.rating_model(
            tf.concat([user_embeddings,location_embeddings], axis=1)
        ),
        self.like_model(
            tf.concat([user_embeddings,location_embeddings], axis=1)
        ),
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

    ratings = features.pop("add","like")
    like = features.pop("like", "add")

    user_embeddings, location_embeddings, rating_predictions, like_predictions = self(features)

    # We compute the loss for each task.
    rating_loss = self.rating_task(
        labels=ratings,
        predictions=rating_predictions,
    )

    like_loss = self.like_task(
        labels=like,
        predictions=like_predictions,
    )
    retrieval_loss = self.retrieval_task(user_embeddings, location_embeddings)

    # And combine them using the loss weights.
    return (self.rating_weight * rating_loss
            + self.retrieval_weight * retrieval_loss + like_loss*self.like_weight)