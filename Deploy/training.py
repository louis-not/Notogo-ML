
def train_recommender():
    """Function to train model for all user_id & location_id in userfeatures
    input  : 
    - userfeatures --> user_id & location_id interaction
    - wishEmbedding --> candidate tower
    output : model 
    """
    import tensorflow_datasets as tfds

    # setup input data pipeline
    wishEmbeddingDs = tfds.load('Wishembedding',split='train')
    locations = wishEmbeddingDs.map(lambda x: x["location_id"])
    print(len(locations))
    userFeaturesDs = tfds.load('Userfeatures',split='train')
    queries = userFeaturesDs.map(lambda x: x["user_id"])
    print(len(queries))
