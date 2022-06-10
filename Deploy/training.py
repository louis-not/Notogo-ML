
def train_recommender():
    """Function to train model for all user_id & location_id in userfeatures
    input  : 
    - userfeatures --> user_id & location_id interaction
    - wishEmbedding --> candidate tower
    output : model 
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import tensorflow_recommenders as tfrs
    from NoToGoModel import NoToGoModel
    import os

    import userFeatures
    import wishEmbedding

    print("Training Begin")
    # Training Data
    builder = tfds.builder('Wishembedding')
    wishEmbeddingDs = tfds.load('Wishembedding',split='train')
    
    builder = tfds.builder('Userfeatures')
    userFeatureDs = tfds.load('Userfeatures',split='train')
    # queries = userFeatureDs.map(lambda x: x["user_id"])
    # print(f"Num of data: {len(queries)}")

    locations = wishEmbeddingDs.map(lambda x: x["location_id"])
    ratings = userFeatureDs.map(lambda x: {
        "location_id": x["location_id"],
        "user_id": x["user_id"],
        "add" : x["add"],
        "like" : x['like']
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
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.09478))       

    model.fit(cached_train, epochs=8)
    metrics = model.evaluate(cached_test, return_dict=True)

    print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
    print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")
    model.evaluate(cached_test, return_dict=True)

    # brute force
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model, k = 20)
    # recommends movies out of the entire movies dataset.
    index.index_from_dataset(
        tf.data.Dataset.zip((locations.batch(100), locations.batch(100).map(model.location_model)))
    )
    _, titles = index(tf.constant(["0"]))
    # print(f"Recommendations for user 0: {titles[0, :3]}")

    # save model 
    tmp = os.getcwd()
    path = os.path.join(tmp, "Deploy")
    path = os.path.join(path, "model")
    print(path)

    # Save the index.
    tf.saved_model.save(index, path)
    print(f"Model saved to {path}")

    return path 