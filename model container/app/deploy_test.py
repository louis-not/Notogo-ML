from pyexpat import model
from Deploy_Retrieval_V2_2 import preprocess_dataset, generate_model_all, inference

 
def main():
    cached_train, cached_test, _, _, locations = preprocess_dataset()
    train_result, test_result, model = generate_model_all(cached_train, cached_test)
    result = inference(model, locations, user_id = "1", limit = 20)
    # print(train_result)
    
if __name__ == '__main__':
    main()