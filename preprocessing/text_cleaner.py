import re
import nltk

from nltk.corpus import stopwords


# Download stopwords
nltk.download("stopwords", quiet=True)


# ==================================================
# STOPWORDS
# ==================================================

stop_words = set(
    stopwords.words("english")
)


# ==================================================
# IMPORTANT NEGATION WORDS
# DO NOT REMOVE THESE
# ==================================================

negation_words = {

    "no",

    "not",

    "never",

    "neither",

    "nor",

    "don't",

    "doesn't",

    "didn't",

    "isn't",

    "wasn't",

    "weren't",

    "can't",

    "couldn't",

    "won't",

    "wouldn't",

    "shouldn't",

    "cannot"

}


# Remove normal stopwords,
# but preserve negation words

stop_words = stop_words - negation_words


# ==================================================
# TEXT CLEANING FUNCTION
# ==================================================

def clean_text(text):


    # ----------------------------------------------
    # Convert to string
    # ----------------------------------------------

    text = str(text)


    # ----------------------------------------------
    # Convert to lowercase
    # ----------------------------------------------

    text = text.lower()


    # ----------------------------------------------
    # Remove URLs
    # ----------------------------------------------

    text = re.sub(

        r"http\S+|www\S+",

        "",

        text

    )


    # ----------------------------------------------
    # Remove mentions
    # ----------------------------------------------

    text = re.sub(

        r"@\w+",

        "",

        text

    )


    # ----------------------------------------------
    # Remove hashtag symbol
    # But keep the word
    #
    # #amazing
    # becomes
    # amazing
    # ----------------------------------------------

    text = re.sub(

        r"#",

        "",

        text

    )


    # ----------------------------------------------
    # Expand common contractions
    #
    # don't → do not
    # doesn't → does not
    # didn't → did not
    # isn't → is not
    # wasn't → was not
    # can't → can not
    # won't → will not
    # ----------------------------------------------

    contractions = {

        "don't": "do not",

        "doesn't": "does not",

        "didn't": "did not",

        "isn't": "is not",

        "wasn't": "was not",

        "weren't": "were not",

        "can't": "can not",

        "couldn't": "could not",

        "won't": "will not",

        "wouldn't": "would not",

        "shouldn't": "should not"

    }


    for contraction, replacement in contractions.items():


        text = text.replace(

            contraction,

            replacement

        )


    # ----------------------------------------------
    # Keep only letters and spaces
    # ----------------------------------------------

    text = re.sub(

        r"[^a-zA-Z\s]",

        "",

        text

    )


    # ----------------------------------------------
    # Split into words
    # ----------------------------------------------

    words = text.split()


    # ----------------------------------------------
    # Remove stopwords
    #
    # But keep:
    # not
    # no
    # never
    # etc.
    # ----------------------------------------------

    words = [

        word

        for word in words

        if word not in stop_words

    ]


    # ----------------------------------------------
    # Join words
    # ----------------------------------------------

    cleaned_text = " ".join(

        words

    )


    return cleaned_text