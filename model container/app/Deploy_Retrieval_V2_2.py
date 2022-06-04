import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import pandas as pd

import tensorflow_recommenders as tfrs


import userFeatures
import wishEmbedding

# import NotogoModel

class NotogoModel(tfrs.models.Model):

  def __init__(self, 
                rating_weight: float, 
                like_weight: float,
                retrieval_weight: float
                ) -> None:

    # We take the loss weights in the constructor: this allows us to instantiate
    # several model objects with different loss weights.
    self.BATCH_SIZE = 180
    self.EMB_DIM = 41

    cached_train, cached_test, unique_location_id, unique_user_ids, locations = preprocess_dataset()

    super().__init__()

    embedding_dimension = int(self.EMB_DIM)

    # User and movie models.
    self.location_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_location_id, mask_token=None),
      tf.keras.layers.Embedding(len(unique_location_id) + 1, embedding_dimension),
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
    location_embeddings = self.location_model(features["location_id"])
    
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


def preparing_dataset():
    builder = tfds.builder('Userfeatures')
    userFeatureDs = tfds.load('Userfeatures',split='train')

    builder = tfds.builder('Wishembedding')
    wishEmbeddingDs = tfds.load('Wishembedding',split='train')

    ratings = userFeatureDs.map(lambda x: {
    "location_id": x["location_id"],
    "user_id": x["user_id"],
    "add" : x["add"],
    "like" : x['like']
    })

    locations = wishEmbeddingDs.map(lambda x: x["location_id"])

    return userFeatureDs, wishEmbeddingDs, ratings, locations


def split_data(ratings, train_size=0.8,seed=42):
    """function to split data, with changeable seed or train percentage"""
    tf.random.set_seed(seed)  # change seed

    NUM_DATA = ratings.__len__().numpy()
    shuffled = ratings.shuffle(NUM_DATA, seed=42, reshuffle_each_iteration=False)
    trainset_size = train_size * NUM_DATA
    train = shuffled.take(trainset_size)
    test = shuffled.skip(trainset_size).take(NUM_DATA - trainset_size)

    cached_train = train.shuffle(NUM_DATA).batch(512).cache()
    cached_test = test.batch(256).cache()

    return train, test, cached_train, cached_test


def preprocess_dataset():
    userFeatureDs, wishEmbeddingDs, ratings, locations = preparing_dataset()
    train, test, cached_train, cached_test = split_data(ratings)
    location_id = locations.batch(1000)
    user_ids = ratings.batch(1000).map(lambda x: x["user_id"])
    unique_location_id = np.unique(np.concatenate(list(location_id)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))
    return cached_train, cached_test, unique_location_id, unique_user_ids, locations


def initialize_model(learning_rate):
    # Initialize Model
    model = NotogoModel(rating_weight=0.0, like_weight = 0, retrieval_weight=1.0)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate))

    return model


def train_model(model, epoch, cached_train):
    # Training Model
    model.fit(cached_train, epochs=epoch)
    result = model.evaluate(cached_train, return_dict=True) 
    return result 


def evaluate_model(model, cached_test):
    # Testing model 
    return model.evaluate(cached_test, return_dict=True)


def generate_model_all(cached_train, cached_test):
    LEARNING_RATE = 0.09478       
    EPOCH = 8
    
    model = initialize_model(LEARNING_RATE)
    print("Model initialized")
    print("Training model")
    train_result = train_model(model, EPOCH, cached_train)
    print("evaluating model")
    test_result = evaluate_model(model, cached_test)

    return train_result, test_result, model


def inference(model, locations, user_id, limit=10):
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model, k = limit)
    # recommends movies out of the entire movies dataset.
    index.index_from_dataset(
    tf.data.Dataset.zip((locations.batch(100), locations.batch(100).map(model.location_model)))
    )

    _, titles = index(tf.constant([user_id]))
    result = titles[0, :limit]
    print(f"Recommendations for New User : {titles[0, :limit]}")

    return result
