

def recommend(model_path: str, user_id: str, limit: int):
    from unittest import result
    import tensorflow as tf
    user_id = str(user_id)
    # result = list()
    loaded = tf.saved_model.load(model_path)

    # Pass a user id in, get top predicted movie titles back.
    scores, result = loaded([user_id])

    # print(f"Recommendations: {result[0][:limit]}")

    return result