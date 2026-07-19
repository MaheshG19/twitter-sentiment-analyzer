import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing.text_cleaner import clean_text


# ==================================================
# DATASET PATHS
# ==================================================

DATASET_PATHS = [

    "data/twitter_sentiment_1000.csv",

    "data/social_media_sentiment_1500.csv",

    "data/balanced_sentiment_3000.csv",

    "data/positive_negative_neutral_5000.csv",

    "data/mixed_social_sentiment_10000.csv",

    "data/twitter_sentiment1000.csv",

    "data/keyword_sentiment_dataset_1000.csv",

    "data/positive_good_dataset_1000.csv",

    "data/neutral_decent_dataset_1000.csv",

    "data/negative_bad_not_dataset_1000.csv",

    "data/mixed_context_sentiment_dataset_1500.csv"






]


# ==================================================
# POSSIBLE COLUMN NAMES
# ==================================================

TEXT_COLUMNS = [

    "clean_text",

    "text",

    "tweet",

    "tweet_text",

    "SentimentText",

    "sentiment_text",

    "content",

    "message"

]


SENTIMENT_COLUMNS = [

    "category",

    "sentiment",

    "Sentiment",

    "label",

    "polarity",

    "emotion"

]


# ==================================================
# FIND COLUMN
# ==================================================

def find_column(df, possible_columns):


    column_map = {

        str(column).lower().strip(): column

        for column in df.columns

    }


    for column in possible_columns:


        if column.lower() in column_map:


            return column_map[

                column.lower()

            ]


    return None


# ==================================================
# NORMALIZE SENTIMENT
# ==================================================

def normalize_sentiment(value):


    if pd.isna(value):

        return None


    value = str(value).strip().lower()


    # ----------------------------------------------
    # NUMERIC SENTIMENT LABELS
    # ----------------------------------------------

    if value in [

        "0",

        "0.0"

    ]:

        return "negative"


    if value in [

        "4",

        "4.0"

    ]:

        return "positive"


    if value in [

        "1",

        "1.0"

    ]:

        return "positive"


    if value in [

        "2",

        "2.0"

    ]:

        return "neutral"


    # ----------------------------------------------
    # TEXT LABELS
    # ----------------------------------------------

    if value in [

        "negative",

        "neg",

        "bad",

        "hate",

        "sad"

    ]:

        return "negative"


    if value in [

        "positive",

        "pos",

        "good",

        "happy",

        "love"

    ]:

        return "positive"


    if value in [

        "neutral",

        "neu",

        "okay",

        "ok"

    ]:

        return "neutral"


    return None


# ==================================================
# LOAD DATASETS
# ==================================================

all_datasets = []


print()

print("=" * 60)

print("LOADING DATASETS")

print("=" * 60)


for dataset_path in DATASET_PATHS:


    print()

    print(

        f"Loading: {dataset_path}"

    )


    try:


        df = pd.read_csv(

            dataset_path,

            encoding="latin1"

        )


        print(

            f"Rows loaded: "

            f"{len(df):,}"

        )


        # ------------------------------------------
        # FIND TEXT COLUMN
        # ------------------------------------------

        text_column = find_column(

            df,

            TEXT_COLUMNS

        )


        # ------------------------------------------
        # FIND SENTIMENT COLUMN
        # ------------------------------------------

        sentiment_column = find_column(

            df,

            SENTIMENT_COLUMNS

        )


        if text_column is None:


            print(

                "⚠️ Text column not found."

            )


            print(

                "Available columns:",

                list(df.columns)

            )


            continue


        if sentiment_column is None:


            print(

                "⚠️ Sentiment column not found."

            )


            print(

                "Available columns:",

                list(df.columns)

            )


            continue


        print(

            f"Text column: "

            f"{text_column}"

        )


        print(

            f"Sentiment column: "

            f"{sentiment_column}"

        )


        # ------------------------------------------
        # STANDARDIZE COLUMNS
        # ------------------------------------------

        standardized_df = pd.DataFrame()


        standardized_df[

            "text"

        ] = df[

            text_column

        ]


        standardized_df[

            "sentiment"

        ] = df[

            sentiment_column

        ]


        # ------------------------------------------
        # NORMALIZE LABELS
        # ------------------------------------------

        standardized_df[

            "category"

        ] = standardized_df[

            "sentiment"

        ].apply(

            normalize_sentiment

        )


        # ------------------------------------------
        # REMOVE INVALID ROWS
        # ------------------------------------------

        standardized_df = standardized_df.dropna(

            subset=[

                "text",

                "category"

            ]

        )


        # ------------------------------------------
        # KEEP ONLY VALID SENTIMENTS
        # ------------------------------------------

        standardized_df = standardized_df[

            standardized_df[

                "category"

            ].isin(

                [

                    "positive",

                    "negative",

                    "neutral"

                ]

            )

        ]


        all_datasets.append(

            standardized_df[

                [

                    "text",

                    "category"

                ]

            ]

        )


        print(

            "✅ Dataset processed successfully."

        )


    except FileNotFoundError:


        print(

            f"⚠️ File not found: "

            f"{dataset_path}"

        )


    except Exception as error:


        print(

            f"❌ Error loading dataset: "

            f"{error}"

        )


# ==================================================
# COMBINE DATASETS
# ==================================================

if len(all_datasets) == 0:


    raise FileNotFoundError(

        "No valid datasets were found."

    )


df = pd.concat(

    all_datasets,

    ignore_index=True

)


print()

print("=" * 60)

print("DATASETS COMBINED")

print("=" * 60)


print(

    f"Total rows: "

    f"{len(df):,}"

)


# ==================================================
# REMOVE MISSING VALUES
# ==================================================

df = df.dropna(

    subset=[

        "text",

        "category"

    ]

)


# ==================================================
# REMOVE EMPTY TEXT
# ==================================================

df["text"] = (

    df["text"]

    .astype(str)

    .str.strip()

)


df = df[

    df["text"].str.len() > 0

]


# ==================================================
# REMOVE DUPLICATES
# ==================================================

before_duplicates = len(df)


df = df.drop_duplicates(

    subset=[

        "text",

        "category"

    ]

)


after_duplicates = len(df)


print()

print(

    "Duplicate rows removed: "

    f"{before_duplicates - after_duplicates:,}"

)


print(

    "Final dataset size: "

    f"{len(df):,}"

)


# ==================================================
# SENTIMENT DISTRIBUTION
# ==================================================

print()

print("=" * 60)

print("SENTIMENT DISTRIBUTION")

print("=" * 60)


print(

    df["category"].value_counts()

)


# ==================================================
# CLEAN TEXT
# ==================================================

print()

print(

    "Cleaning tweets..."

)


df["cleaned_tweet"] = (

    df["text"]

    .astype(str)

    .apply(clean_text)

)


# ==================================================
# REMOVE EMPTY CLEANED TEXT
# ==================================================

df = df[

    df["cleaned_tweet"]

    .str.strip()

    .str.len()

    > 0

]


# ==================================================
# INPUT AND TARGET
# ==================================================

X = df["cleaned_tweet"]

y = df["category"]


# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)


print()

print(

    f"Training samples: "

    f"{len(X_train):,}"

)


print(

    f"Testing samples: "

    f"{len(X_test):,}"

)


# ==================================================
# TF-IDF
# ==================================================

print()

print(

    "Creating TF-IDF features..."

)


tfidf = TfidfVectorizer(

    max_features=10000,

    ngram_range=(1, 2),

    min_df=1,

    max_df=0.95,

    sublinear_tf=True

)


X_train_tfidf = tfidf.fit_transform(

    X_train

)


X_test_tfidf = tfidf.transform(

    X_test

)


print()

print(

    "TF-IDF feature count: "

    f"{X_train_tfidf.shape[1]:,}"

)


# ==================================================
# TRAIN MODEL
# ==================================================

print()

print(

    "Training Logistic Regression..."

)


model = LogisticRegression(

    max_iter=1000,

    class_weight="balanced"

)


model.fit(

    X_train_tfidf,

    y_train

)


# ==================================================
# PREDICTION
# ==================================================

y_pred = model.predict(

    X_test_tfidf

)


# ==================================================
# METRICS
# ==================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)


precision = precision_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0

)


recall = recall_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0

)


f1 = f1_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0

)


# ==================================================
# DISPLAY RESULTS
# ==================================================

print()

print("=" * 60)

print("MODEL PERFORMANCE")

print("=" * 60)


print(

    f"Accuracy : "

    f"{accuracy * 100:.2f}%"

)


print(

    f"Precision: "

    f"{precision * 100:.2f}%"

)


print(

    f"Recall   : "

    f"{recall * 100:.2f}%"

)


print(

    f"F1 Score : "

    f"{f1 * 100:.2f}%"

)


print("=" * 60)


# ==================================================
# CLASSIFICATION REPORT
# ==================================================

print()

print(

    classification_report(

        y_test,

        y_pred

    )

)


# ==================================================
# CONFUSION MATRIX
# ==================================================

print()

print(

    "Confusion Matrix:"

)


print(

    confusion_matrix(

        y_test,

        y_pred

    )

)


# ==================================================
# SAVE MODEL
# ==================================================

os.makedirs(

    "models",

    exist_ok=True

)


joblib.dump(

    model,

    "models/sentiment_model.pkl"

)


joblib.dump(

    tfidf,

    "models/tfidf_vectorizer.pkl"

)


# ==================================================
# SAVE METRICS
# ==================================================

metrics = {

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1_score": f1

}


joblib.dump(

    metrics,

    "models/metrics.pkl"

)


# ==================================================
# SAVE TEST RESULTS
# ==================================================

results = pd.DataFrame(

    {

        "actual": y_test.values,

        "predicted": y_pred

    }

)


results.to_csv(

    "models/test_results.csv",

    index=False

)


# ==================================================
# FINISHED
# ==================================================

print()

print("=" * 60)

print(

    "✅ MODEL TRAINING COMPLETED!"

)

print("=" * 60)


print()

print(

    "Model saved at:"

)


print(

    "models/sentiment_model.pkl"

)


print()

print(

    "Vectorizer saved at:"

)


print(

    "models/tfidf_vectorizer.pkl"

)