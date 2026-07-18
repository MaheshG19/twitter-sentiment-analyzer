import streamlit as st
import pandas as pd
import plotly.express as px

from utils.dataset_loader import (
    load_dataset,
    find_text_column
)

from preprocessing.text_cleaner import clean_text

from ml.predict import (
    predict_sentiment,
    predict_batch
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(

    page_title="Twitter Sentiment Analyzer",

    page_icon="🐦",

    layout="wide"

)


# ==================================================
# HEADER
# ==================================================

st.title(

    "🐦 Twitter Sentiment Analyzer"

)

st.write(

    "Analyze tweets using Natural Language Processing "

    "and Machine Learning."

)

st.divider()


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title(

    "⚙️ Analysis Options"

)


option = st.sidebar.radio(

    "Choose Analysis Mode",

    [

        "📝 Analyze Single Tweet",

        "📂 Upload CSV Dataset"

    ]

)


# ==================================================
# SINGLE TWEET ANALYSIS
# ==================================================

if option == "📝 Analyze Single Tweet":


    st.header(

        "📝 Analyze a Single Tweet"

    )


    tweet = st.text_area(

        "Enter your tweet",

        placeholder=

        "Example: I absolutely love this movie! 😍",

        height=150

    )


    if st.button(

        "🔍 Analyze Sentiment",

        use_container_width=True

    ):


        if not tweet.strip():


            st.warning(

                "⚠️ Please enter a tweet."

            )

            st.stop()


        # ------------------------------------------
        # CLEAN TEXT
        # ------------------------------------------

        cleaned_tweet = clean_text(

            tweet

        )
        st.write("Original text:", tweet)
        st.write("Cleaned text:", cleaned_tweet)


        # ------------------------------------------
        # PREDICT SENTIMENT
        # ------------------------------------------

        sentiment, confidence, probabilities = (

            predict_sentiment(

                cleaned_tweet

            )

        )


        sentiment_lower = (

            sentiment.lower()

        )


        # ------------------------------------------
        # EMOJI
        # ------------------------------------------

        if sentiment_lower == "positive":

            emoji = "😊"

        elif sentiment_lower == "negative":

            emoji = "😠"

        else:

            emoji = "😐"


        st.divider()


        # ------------------------------------------
        # RESULT CARDS
        # ------------------------------------------

        col1, col2 = st.columns(2)


        with col1:


            st.metric(

                "Predicted Sentiment",

                f"{emoji} {sentiment.title()}"

            )


        with col2:


            st.metric(

                "Confidence",

                f"{confidence * 100:.2f}%"

            )


        # ------------------------------------------
        # RESULT MESSAGE
        # ------------------------------------------

        if sentiment_lower == "positive":


            st.success(

                "😊 This tweet is Positive"

            )


        elif sentiment_lower == "negative":


            st.error(

                "😠 This tweet is Negative"

            )


        else:


            st.info(

                "😐 This tweet is Neutral"

            )


        # ------------------------------------------
        # PROBABILITY CHART
        # ------------------------------------------

        st.subheader(

            "📈 Sentiment Probabilities"

        )


        probability_df = pd.DataFrame(

            {

                "Sentiment":

                list(

                    probabilities.keys()

                ),

                "Probability":

                [

                    value * 100

                    for value

                    in probabilities.values()

                ]

            }

        )


        fig = px.bar(

            probability_df,

            x="Sentiment",

            y="Probability",

            text="Probability",

            title="Model Confidence"

        )


        st.plotly_chart(

            fig,

            use_container_width=True

        )


# ==================================================
# CSV UPLOAD AND BATCH ANALYSIS
# ==================================================

else:


    st.header(

        "📂 Upload CSV Dataset"

    )


    st.write(

        "Upload a CSV file to analyze multiple tweets."

    )


    # ----------------------------------------------
    # FILE UPLOADER
    # ----------------------------------------------

    uploaded_file = st.file_uploader(

        "Choose a CSV file",

        type=["csv"]

    )


    if uploaded_file is None:


        st.info(

            "👈 Upload a CSV file to start analysis."

        )


    else:


        # ==========================================
        # LOAD DATASET
        # ==========================================

        try:


            df = load_dataset(

                uploaded_file

            )


            st.success(

                "✅ Dataset loaded successfully!"

            )


        except Exception as e:


            st.error(

                f"❌ Error loading file: {e}"

            )

            st.stop()


        # ==========================================
        # FIND TEXT COLUMN
        # ==========================================

        text_column = find_text_column(

            df

        )


        if text_column is None:


            st.error(

                "❌ Could not detect a text column."

            )


            st.write(

                "Available columns:"

            )


            st.write(

                list(df.columns)

            )


            st.stop()


        st.success(

            f"📝 Text column detected: "

            f"`{text_column}`"

        )


        # ==========================================
        # DATASET OVERVIEW
        # ==========================================

        st.subheader(

            "📊 Dataset Overview"

        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:


            st.metric(

                "Total Rows",

                f"{len(df):,}"

            )


        with col2:


            st.metric(

                "Total Columns",

                len(df.columns)

            )


        with col3:


            st.metric(

                "Missing Text",

                df[text_column]

                .isna()

                .sum()

            )


        with col4:


            st.metric(

                "Duplicate Rows",

                df.duplicated()

                .sum()

            )


        # ==========================================
        # DATASET PREVIEW
        # ==========================================

        st.subheader(

            "👀 Dataset Preview"

        )


        st.dataframe(

            df.head(20),

            use_container_width=True

        )


        # ==========================================
        # PREDICTION BUTTON
        # ==========================================

        if st.button(

            "🤖 Predict All Tweet Sentiments",

            use_container_width=True

        ):


            with st.spinner(

                "🔄 Analyzing tweets..."

            ):


                # ----------------------------------
                # PREPARE TEXT
                # ----------------------------------

                if text_column.lower() in [

                    "clean_text",

                    "cleaned_text",

                    "cleaned_tweet"

                ]:


                    cleaned_tweets = (

                        df[text_column]

                        .fillna("")

                        .astype(str)

                        .tolist()

                    )


                else:


                    cleaned_tweets = (

                        df[text_column]

                        .fillna("")

                        .astype(str)

                        .apply(clean_text)

                        .tolist()

                    )


                # ----------------------------------
                # BATCH PREDICTION
                # ----------------------------------

                predictions = predict_batch(

                    cleaned_tweets

                )


                # ----------------------------------
                # ADD PREDICTED SENTIMENT
                # ----------------------------------

                df[

                    "predicted_sentiment"

                ] = predictions[

                    "predicted_sentiment"

                ]


                # ----------------------------------
                # ADD CONFIDENCE
                # ----------------------------------

                df[

                    "confidence"

                ] = (

                    predictions[

                        "confidence"

                    ]

                    * 100

                ).round(2)


                # ----------------------------------
                # ADD PROBABILITY COLUMNS
                # ----------------------------------

                for column in predictions.columns:


                    if column not in [

                        "predicted_sentiment",

                        "confidence"

                    ]:


                        df[column] = (

                            predictions[column]

                            * 100

                        ).round(2)


                # ----------------------------------
                # SAVE RESULTS
                # ----------------------------------

                st.session_state[

                    "prediction_results"

                ] = df


                st.success(

                    "✅ Sentiment prediction completed!"

                )


        # ==========================================
        # SHOW ANALYTICS
        # ==========================================

        if (

            "prediction_results"

            in st.session_state

        ):


            result_df = st.session_state[

                "prediction_results"

            ].copy()


            st.divider()


            st.header(

                "📊 Sentiment Analytics"

            )


            # ======================================
            # SEARCH AND FILTER
            # ======================================

            st.subheader(

                "🔍 Search and Filter"

            )


            col1, col2 = st.columns(2)


            with col1:


                search_text = st.text_input(

                    "🔎 Search tweets",

                    placeholder=

                    "Search for words..."

                )


            with col2:


                sentiment_options = [

                    "All"

                ] + sorted(

                    result_df[

                        "predicted_sentiment"

                    ]

                    .dropna()

                    .unique()

                    .tolist()

                )


                selected_sentiment = st.selectbox(

                    "🎯 Filter by sentiment",

                    sentiment_options

                )


            # ======================================
            # APPLY SEARCH FILTER
            # ======================================

            filtered_df = result_df.copy()


            if search_text.strip():


                filtered_df = filtered_df[

                    filtered_df[

                        text_column

                    ]

                    .astype(str)

                    .str.contains(

                        search_text,

                        case=False,

                        na=False

                    )

                ]


            # ======================================
            # APPLY SENTIMENT FILTER
            # ======================================

            if selected_sentiment != "All":


                filtered_df = filtered_df[

                    filtered_df[

                        "predicted_sentiment"

                    ]

                    == selected_sentiment

                ]


            st.write(

                f"Showing "

                f"**{len(filtered_df):,}** "

                f"of "

                f"**{len(result_df):,}** rows"

            )


            # ======================================
            # FILTERED DATA
            # ======================================

            st.dataframe(

                filtered_df.head(100),

                use_container_width=True

            )


            # ======================================
            # KEY STATISTICS
            # ======================================

            st.subheader(

                "📌 Key Statistics"

            )


            sentiment_counts = (

                result_df[

                    "predicted_sentiment"

                ]

                .value_counts()

            )


            col1, col2, col3, col4 = st.columns(4)


            with col1:


                st.metric(

                    "Total Tweets",

                    f"{len(result_df):,}"

                )


            with col2:


                st.metric(

                    "Positive",

                    sentiment_counts.get(

                        "positive",

                        0

                    )

                )


            with col3:


                st.metric(

                    "Negative",

                    sentiment_counts.get(

                        "negative",

                        0

                    )

                )


            with col4:


                st.metric(

                    "Neutral",

                    sentiment_counts.get(

                        "neutral",

                        0

                    )

                )


            # ======================================
            # SENTIMENT DISTRIBUTION
            # ======================================

            st.subheader(

                "📈 Sentiment Distribution"

            )


            distribution_df = (

                result_df[

                    "predicted_sentiment"

                ]

                .value_counts()

                .reset_index()

            )


            distribution_df.columns = [

                "Sentiment",

                "Count"

            ]


            col1, col2 = st.columns(2)


            with col1:


                fig_bar = px.bar(

                    distribution_df,

                    x="Sentiment",

                    y="Count",

                    text="Count",

                    title=

                    "Sentiment Count"

                )


                st.plotly_chart(

                    fig_bar,

                    use_container_width=True

                )


            with col2:


                fig_pie = px.pie(

                    distribution_df,

                    names="Sentiment",

                    values="Count",

                    title=

                    "Sentiment Percentage"

                )


                st.plotly_chart(

                    fig_pie,

                    use_container_width=True

                )


            # ======================================
            # CONFIDENCE ANALYSIS
            # ======================================

            st.subheader(

                "🎯 Confidence Analysis"

            )


            col1, col2, col3 = st.columns(3)


            with col1:


                st.metric(

                    "Average Confidence",

                    f"{result_df['confidence'].mean():.2f}%"

                )


            with col2:


                st.metric(

                    "Highest Confidence",

                    f"{result_df['confidence'].max():.2f}%"

                )


            with col3:


                st.metric(

                    "Lowest Confidence",

                    f"{result_df['confidence'].min():.2f}%"

                )


            # ======================================
            # CONFIDENCE HISTOGRAM
            # ======================================

            confidence_fig = px.histogram(

                result_df,

                x="confidence",

                nbins=20,

                title=

                "Confidence Score Distribution"

            )


            st.plotly_chart(

                confidence_fig,

                use_container_width=True

            )


            # ======================================
            # DOWNLOAD RESULTS
            # ======================================

            st.subheader(

                "⬇️ Download Results"

            )


            csv_data = (

                filtered_df

                .to_csv(

                    index=False

                )

                .encode(

                    "utf-8"

                )

            )


            st.download_button(

                label=

                "📥 Download Filtered Results",

                data=csv_data,

                file_name=

                "sentiment_analysis_results.csv",

                mime=

                "text/csv",

                use_container_width=True

            )