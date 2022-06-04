import pandas as pd
import mysql.connector
from wishEmbedding import wishEmbedding
import tensorflow as tf
import tensorflow_datasets as tfds
from userFeatures import userFeatures
import numpy as np
from NoToGoModel import NoToGoModel
import datetime
import os


def main():
    print('Hello World')

    # Training Data
    tfds.builder('Wishembedding')
    wishEmbeddingDs = tfds.load('Wishembedding',split='train')
    
    tfds.builder('Userfeatures')
    userFeatureDs = tfds.load('Userfeatures',split='train')
    queries = userFeatureDs.map(lambda x: x["user_id"])
    print(len(queries))

    locations = wishEmbeddingDs.map(lambda x: x["location_id"])
    ratings = userFeatureDs.map(lambda x: {
        "user_id": x["user_id"],
        "add" : x["add"],
        "like" : x["like"]
    })  

    tf.random.set_seed(42)

    NUM_DATA = ratings.__len__().numpy()

    shuffled = ratings.shuffle(NUM_DATA, seed=42, reshuffle_each_iteration=False)

    trainset_size = 0.8 * NUM_DATA

    train = shuffled.take(trainset_size)
    test = shuffled.skip(trainset_size).take(NUM_DATA - trainset_size)
    cached_train = train.shuffle(NUM_DATA).batch(512).cache()
    cached_test = test.batch(256).cache()

    parameter = {
        'ratings' : ratings,
        'locations' : locations
    }

    model = NoToGoModel(input=parameter, rating_weight=0.0, like_weight = 0, retrieval_weight=1.0)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))       

    model.fit(cached_train, epochs=10)
    metrics = model.evaluate(cached_test, return_dict=True)

    print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
    print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")
    model.evaluate(cached_test, return_dict=True)

if __name__ == '__main__':
    main()
