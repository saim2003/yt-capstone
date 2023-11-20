import streamlit as st
from googleapiclient.discovery import build
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# YouTube Data API v3 setup
api_service_name = "youtube"
api_version = "v3"
api_key = "AIzaSyDi27OID7eh35pvpNFTX6Ui0tspGXl48mY"  # Replace with your YouTube API key

# Create a YouTube service object
youtube = build(api_service_name, api_version, developerKey=api_key)

# Sentiment analysis setup
sentiment_analyzer = pipeline("sentiment-analysis")
text_summarizer = pipeline("summarization")

# Function to fetch video comments using YouTube API
def fetch_video_comments(video_id):
    comments = []

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,  # Fetch top 100 comments
            order="relevance",  # You can modify this as per your preference
        )
        response = request.execute()

        for comment in response["items"]:
            comments.append(comment["snippet"]["topLevelComment"]["snippet"]["textOriginal"])

    except Exception as e:
        st.error(f"Error fetching comments: {str(e)}")

    return comments

# Function to summarize sentiments from comments
def summarize_sentiments(comments):
    positive_comments = []
    neutral_comments = []
    negative_comments = []

    for comment in comments:
        sentiment = sentiment_analyzer(comment)[0]
        if sentiment['label'] == 'POSITIVE':
            positive_comments.append(comment)
        elif sentiment['label'] == 'NEUTRAL':
            neutral_comments.append(comment)
        elif sentiment['label'] == 'NEGATIVE':
            negative_comments.append(comment)

    return {
        'positive': positive_comments,
        'neutral': neutral_comments,
        'negative': negative_comments
    }

# Function to summarize comments in a specified word limit
def summarize_comments(comments, max_words):
    summarized_text = ''
    if len(comments) > 0:
        summarized_text = text_summarizer(' '.join(comments), max_length=max_words, min_length=30, do_sample=False)
        if summarized_text and len(summarized_text) > 0 and 'summary_text' in summarized_text[0]:
            return summarized_text[0]['summary_text']
    return "No comments to summarize within the specified word limit."

def main():
    st.title("TLDR: YouTube Comment Sentiment Analysis")
    st.write("Enter YouTube video ID below:")

    video_id = st.text_input("Paste the YouTube video ID here:")

    if st.button("Analyze"):
        if video_id:
            try:
                comments = fetch_video_comments(video_id)

                if comments:
                    st.subheader("Top 100 Comments:")
                    st.write(comments[:100])  # Display top 100 comments if available

                    summaries = summarize_sentiments(comments)

                    st.subheader("Sentiment Analysis Summary:")
                    st.write(f"Total Comments: {len(comments)}")
                    st.write(f"Positive Comments: {len(summaries['positive'])}")
                    st.write(f"Neutral Comments: {len(summaries['neutral'])}")
                    st.write(f"Negative Comments: {len(summaries['negative'])}")

                    if summaries['positive'] and summaries['negative']:
                        # Plotting sentiment summary as a pie chart
                        labels = ['Positive', 'Neutral', 'Negative']
                        sizes = [len(summaries['positive']), len(summaries['neutral']), len(summaries['negative'])]
                        explode = (0.1, 0, 0)  # Explode the 1st slice (Positive)

                        fig1, ax1 = plt.subplots()
                        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
                        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                        st.subheader("Sentiment Analysis Summary (Pie Chart)")
                        st.pyplot(fig1)

                        # Plot word cloud for comments
                        text = ' '.join(comments)
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')

                        st.subheader("Word Cloud for Comments")
                        st.pyplot(plt)

                        # Display top positive comments
                        st.subheader("Top Positive Comments:")
                        st.write(summaries['positive'][:10])  # Display top 10 positive comments if available

                        # Display top negative comments
                        st.subheader("Top Negative Comments:")
                        st.write(summaries['negative'][:10])  # Display top 10 negative comments if available

                        # Overall comment summarisation in 100 words
                        overall_summary = summarize_comments(comments, 100)
                        st.subheader("Overall Comment Summarization (100 words):")
                        st.write(overall_summary)

                        # Positive comment summarisation in 100 words
                        positive_summary = summarize_comments(summaries['positive'], 100)
                        st.subheader("Positive Comment Summarization (100 words):")
                        st.write(positive_summary)

                        # Negative comment summarisation in 100 words
                        negative_summary = summarize_comments(summaries['negative'], 100)
                        st.subheader("Negative Comment Summarization (100 words):")
                        st.write(negative_summary)

                    else:
                        st.warning("No positive or negative comments found for this video.")

                else:
                    st.warning("No comments found for this video.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
