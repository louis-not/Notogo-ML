from training import train_recommender
from recommender import recommend
import os

from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

class HomePage(Resource):
    def get(self):
        return 'ML API ONLINE', 200

class Recommendation(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('user_id', required=True)

        args = parser.parse_args()
        user_id = args['user_id']
        user_id = str(user_id)
        print(user_id)

        tmp = os.getcwd()
        path = os.path.join(tmp, "Deploy")
        model_path = os.path.join(path, "model")

        limit = 20
        result = recommend(model_path, user_id, limit)

        return {
            'error': False,
            'result': f"{result[0][:limit]}"
        }, 200

class Training(Resource):
    def post(self):
        train_recommender()
        return {
            'error': False,
            'message': 'Training done'
        }

api.add_resource(HomePage, '/')
api.add_resource(Recommendation, '/recommend')
api.add_resource(Training, '/train')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    # train_recommender()
    # tmp = os.getcwd()
    # path = os.path.join(tmp, "Deploy")
    # model_path = os.path.join(path, "model")
    # limit = 20
    # user_id = '0'
    # result = recommend(model_path, user_id, limit)
    # print(f"Recommendations {user_id}: {result[0][:limit]}")

    # # load model
    # limit = 20
    # user_id = '30'
    # result = recommend(model_path, user_id, limit)
    # print(f"Recommendations {user_id}: {result[0][:limit]}")
