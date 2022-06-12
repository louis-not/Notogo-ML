# Notogo-ML
Notogo Recommender System using [Multitask Recommenders System.](https://www.tensorflow.org/recommenders/examples/multitask)

## Dataset 
Google Sheet [here](https://docs.google.com/spreadsheets/d/1WNQAsPDfus-rihmvLX5MbUR8dNTth3rm9YK5-6ekUIY/edit#gid=572466784)
| User Embedding  | Wish Embedding | User Features | Wish Features |
| --------------- | -------------- | ------------- | ------------- |
| user_id         | id             | user_id       | id            |
| user_age        | categories     | user_like     | like_count    |
| user_domicle    | coordinate     | user_add      | add_count     |
| user_occupation |                | location_id   | rating        |


## ML Architecture
We use the Multi-task Recommender model by Tensorflow Recommenders.

<p>This is our model architecture :</p>

<p align="left">
    <img src="contents/model architecture.png" alt="Model Architecture" height="500">
</p>

<p>Multitasking means that our model has multiple objectives. Our model has 3 objectives, I.e predicting whether users will <b>add</b> the recommendation result to their wishlist, predicting whether users will <b>like</b> the recommendation result or not, and giving users the recommendation <b>(retrieval)</b>. </p>

## ML Evaluation
Full Evaluation [here](https://docs.google.com/spreadsheets/d/1WrgL-iTQBquAcbi89h_QD4-CZyZfo9QhmSbdR5TxZ68/edit?usp=sharing)
<br> Last Model :
<br> v2.3 
<br> Top_10_categorical_accuracy (TRAIN): 0.6076
<br> Top_10_categorical_accuracy (TEST) : 0.6393

## ML Deployment
Inside Notogo-ML/Deploy-API we could see the deployment script for this application. 
- `NoToGoModel.py` : contained the recommender system model with as a NotogoModel class
<br> `__init__` : create model architecture and intialize input & output
<br> `call` : pick feature 
<br> `compute_loss` : compute loss on rating_task, like_task and retrieval_task and return the generated loss based on input weight
- `training.py` : contained `train_recommender()` function and would return the saved model path
- `recommender.py` : contained `recommend()` function and would return recommended location_id based on user_id
- `main.py` : Flask API, contained 3 class, i.e `HomePage()`, `Recommendation()`, and `Training()`
- `Dockerfile` : contains all the commands a user could call on the command line to assemble an image, such as specifying the base image, copying all code needed, installing libraries, etc

Here is the following step to deploy the model to the Cloud Run:
- Install the  Google Cloud SDK Shell, you can find the tutorial [Here.](https://www.youtube.com/watch?v=k-8qFh8EfFA)
- Open the Google Cloud SDK Shell and change the directory to the folder that contains all files in the `Notogo-ML/Deploy-API` 
