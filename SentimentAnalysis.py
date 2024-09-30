# -*- coding: utf-8 -*-

# Importing necessary modules
import nltk
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
from nltk import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import string

# Function to download NLTK resources
def download_nltk_resources():
    nltk.download("all")

# Function to generate URLs for review pages
def generate_review_urls(product_id, total_pages=9):
    base_url = f"https://www.amazon.com/{product_id}/product-reviews/{product_id}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
    return [f"{base_url}&pageNumber={i}" for i in range(1, total_pages + 1)]

# Function to scrape reviews and their titles
def scrape_reviews(url):
    params = {'api_key': "ce7562064ab1029af3f879b124f5233a", 'url': url}
    response = requests.get('http://api.scraperapi.com/', params=urlencode(params))
    soup = BeautifulSoup(response.text, 'html.parser')

    # Scrape star rating only from the first page
    product_star_rating = None
    if url == urls[0]:
        item = soup.find("span", {"data-hook": "rating-out-of-text"})
        product_star_rating = item.get_text()

    # Scraping review titles and contents
    titles = {i + 1: item.get_text().strip() for i, item in enumerate(soup.find_all("a", "review-title"))}
    reviews = [item.get_text() for item in soup.find_all("span", {"data-hook": "review-body"})]

    return titles, reviews, product_star_rating

# Function to preprocess reviews
def preprocess_reviews(reviews):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = nltk.stem.WordNetLemmatizer()
    punctuation_free_reviews = [review.translate(str.maketrans('', '', string.punctuation)) for review in reviews]

    tokenized_reviews = [word_tokenize(review) for review in punctuation_free_reviews]
    cleaned_reviews = [[lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words] for tokens in tokenized_reviews]

    return cleaned_reviews

# Function to handle negations
def handle_negations(reviews):
    negated_reviews = []
    for review in reviews:
        temp_review = []
        negation = False
        for word in review:
            if word in ["not", "no", "never"]:
                negation = True
            elif negation:
                temp_review.append("not_" + word)
                if word in [".", "!", "?"]:  # Reset negation at sentence end
                    negation = False
            else:
                temp_review.append(word)
        negated_reviews.append(temp_review)
    return negated_reviews

# Function to load adjective vocabulary
def load_adjective_vocab(file_path):
    with open(file_path) as file:
        return [line.strip() for line in file]

# Function to validate reviews against vocabulary
def validate_reviews(reviews, vocab):
    return [[token for token in review if token in vocab] for review in reviews]

# Function for sentiment analysis
def analyze_sentiment(reviews):
    sia = SentimentIntensityAnalyzer()
    sentiment_counts = {'very_positive': 0, 'positive': 0, 'negative': 0, 'very_negative': 0, 'neutral': 0}

    for review in reviews:
        if not review:
            continue
        score = sum(sia.polarity_scores(w)["compound"] for w in review) / len(review)
        if score > 0.2:
            sentiment_counts['very_positive'] += 1
        elif score > 0.05:
            sentiment_counts['positive'] += 1
        elif score < -0.2:
            sentiment_counts['very_negative'] += 1
        elif score < -0.05:
            sentiment_counts['negative'] += 1
        else:
            sentiment_counts['neutral'] += 1

    return sentiment_counts

# Main function to orchestrate the process
def main():
    download_nltk_resources()

    product_id = "Apple-Generation-Cancelling-Personalized-Customizable/product-reviews/B0BDHWDR12"
    urls = generate_review_urls(product_id)
    product = "Apple AirPods Pro (2nd Generation) Wireless Earbuds".upper()
    print("SENTIMENT ANALYSIS OF", product, "AMAZON'S REVIEWS\n")

    total_sentiment_counts = {'very_positive': 0, 'positive': 0, 'negative': 0, 'very_negative': 0, 'neutral': 0}

    # Process each URL
    for i, url in enumerate(urls):
        print(f"\nPage {i + 1}:")
        titles, reviews, product_star_rating = scrape_reviews(url)
        print("\nTITLES:", titles)

        cleaned_reviews = preprocess_reviews(reviews)
        adj_vocab = load_adjective_vocab("wordnetAdj.txt")
        validated_reviews = validate_reviews(cleaned_reviews, adj_vocab)
        negated_reviews = handle_negations(validated_reviews)

        sentiment_counts = analyze_sentiment(negated_reviews)
        for key in total_sentiment_counts:
            total_sentiment_counts[key] += sentiment_counts[key]

    # Display overall sentiment analysis
    total_reviews = sum(total_sentiment_counts.values())
    print("\nTotal Reviews =", total_reviews)
    print(
        f"Total Very Positive Reviews = {total_sentiment_counts['very_positive']}\tTotal Positive Reviews = {total_sentiment_counts['positive']}")
    print(
        f"Total Very Negative Reviews = {total_sentiment_counts['very_negative']}\tTotal Negative Reviews = {total_sentiment_counts['negative']}")
    print(f"Total Neutral Reviews = {total_sentiment_counts['neutral']}")

    overall_positive_reviews = "{:.4%}".format(
        (total_sentiment_counts['very_positive'] + total_sentiment_counts['positive']) / total_reviews)
    overall_negative_reviews = "{:.4%}".format(
        (total_sentiment_counts['very_negative'] + total_sentiment_counts['negative']) / total_reviews)
    overall_neutral_reviews = "{:.4%}".format(total_sentiment_counts['neutral'] / total_reviews)

    print("\nOverall Positive Reviews:", overall_positive_reviews)
    print("Overall Negative Reviews:", overall_negative_reviews)
    print("Overall Neutral Reviews:", overall_neutral_reviews)
    print("\nCompare to the product's star rating of", product_star_rating)

# Run the main function
if __name__ == "__main__":
    main()

