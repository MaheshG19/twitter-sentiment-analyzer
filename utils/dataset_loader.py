import pandas as pd


# ==================================================
# LOAD CSV DATASET
# ==================================================

def load_dataset(file):

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1"
    ]

    last_error = None

    for encoding in encodings:

        try:

            file.seek(0)

            df = pd.read_csv(
                file,
                encoding=encoding
            )

            return df

        except UnicodeDecodeError as error:

            last_error = error

            continue

    raise ValueError(
        "Unable to read this CSV file. "
        "The file encoding is not supported. "
        f"Last error: {last_error}"
    )


# ==================================================
# FIND TEXT COLUMN
# ==================================================

def find_text_column(df):

    possible_columns = [

        "tweet",

        "text",

        "content",

        "message",

        "tweet_text",

        "tweet_texts",

        "sentimenttext",

        "sentiment_text",

        "clean_text",

        "cleaned_text",

        "cleaned_tweet"

    ]


    # Convert actual column names to lowercase

    column_map = {

        str(column).lower().strip(): column

        for column in df.columns

    }


    # Find column

    for column in possible_columns:

        if column.lower() in column_map:

            return column_map[

                column.lower()

            ]


    return None


# ==================================================
# FIND SENTIMENT COLUMN
# ==================================================

def find_sentiment_column(df):

    possible_columns = [

        "sentiment",

        "label",

        "category",

        "polarity",

        "emotion"

    ]


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