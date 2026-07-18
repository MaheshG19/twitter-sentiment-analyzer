import joblib
import pandas as pd


# ==================================================
# LOAD MODEL
# ==================================================

MODEL_PATH = "models/sentiment_model.pkl"

VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


model = joblib.load(

    MODEL_PATH

)


tfidf = joblib.load(

    VECTORIZER_PATH

)


# ==================================================
# SINGLE TWEET PREDICTION
# ==================================================

def predict_sentiment(

    cleaned_tweet

):

    # Convert text to TF-IDF

    tweet_vector = tfidf.transform(

        [cleaned_tweet]

    )


    # Predict sentiment

    prediction = model.predict(

        tweet_vector

    )[0]


    # Get probabilities

    probabilities_array = (

        model.predict_proba(

            tweet_vector

        )[0]

    )


    # Get class names

    classes = model.classes_


    probabilities = {

        class_name: probability

        for class_name, probability

        in zip(

            classes,

            probabilities_array

        )

    }


    # Highest probability

    confidence = max(

        probabilities_array

    )


    return (

        prediction,

        confidence,

        probabilities

    )


# ==================================================
# BATCH PREDICTION
# ==================================================

def predict_batch(

    cleaned_tweets

):

    # Convert tweets to TF-IDF

    tweet_vectors = tfidf.transform(

        cleaned_tweets

    )


    # Predict sentiments

    predictions = model.predict(

        tweet_vectors

    )


    # Get probabilities

    probabilities = model.predict_proba(

        tweet_vectors

    )


    # Get model classes

    classes = model.classes_


    # Get maximum probability

    confidence = probabilities.max(

        axis=1

    )


    # Create result dataframe

    results = pd.DataFrame(

        {

            "predicted_sentiment":

            predictions,


            "confidence":

            confidence

        }

    )


    # Add probability for each class

    for index, class_name in enumerate(

        classes

    ):

        results[

            f"{class_name}_probability"

        ] = (

            probabilities[:, index]

        )


    return results