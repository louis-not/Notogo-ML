from training import train_recommender
from recommender import recommend
import tensorflow as tf
import os

def main():
    print('Example Use case')

    # train model
    tmp = os.getcwd()
    path = os.path.join(tmp, "Deploy")
    model_path = os.path.join(path, "model")
    model_path = train_recommender()  # comment this line to test recommend()
    # print(model_path)

    # load model
    limit = 20
    user_id = '0'
    result = recommend(model_path, user_id, limit)

    print(f"Recommendations {user_id}: {result[0][:limit]}")

if __name__ == '__main__':
    main()
