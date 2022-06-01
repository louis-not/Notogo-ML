import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import pandas as pd

import tensorflow_recommenders as tfrs

# getting data
import gspread
from google.auth import default
from gspread_dataframe import get_as_dataframe, set_with_dataframe

import userFeatures
import wishEmbedding

import NotogoModel


def preparing_dataset():
    builder = tfds.builder('Userfeatures')
    userFeatureDs = tfds.load('Userfeatures',split='train')

    builder = tfds.builder('Wishembedding')
    wishEmbeddingDs = tfds.load('Wishembedding',split='train')

    ratings = userFeatureDs.map(lambda x: {
    "location_name": x["location_name"],
    "user_id": x["user_id"],
    "add" : x["add"],
    "like" : x['like']
    })

    locations = wishEmbeddingDs.map(lambda x: x["location_name"])

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
    location_name = locations.batch(1000)
    user_ids = ratings.batch(1000).map(lambda x: x["user_id"])
    unique_location_name = np.unique(np.concatenate(list(location_name)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))
    return cached_train, cached_test, unique_location_name, unique_user_ids, locations


def initialize_model(learning_rate):
    # Initialize Model
    model = NotogoModel(rating_weight=0.0, like_weight = 0, retrieval_weight=1.0)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate))

    return model


def train_model(model, epoch, cached_train):
    # Training Model
    model.fit(cached_train, epoch=epoch)
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
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    # recommends movies out of the entire movies dataset.
    index.index_from_dataset(
    tf.data.Dataset.zip((locations.batch(100), locations.batch(100).map(model.location_model)))
    )

    _, titles = index(tf.constant([user_id]))
    result = titles[0, :limit]
    print(f"Recommendations for New User : {titles[0, :limit]}")

    return result