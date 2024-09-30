# -*- coding: utf-8 -*-

# Importing necessary modules
import nltk
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
from nltk import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import string

# Class for managing the review scraping process
class ReviewScraper:
    def __init__(self, product_id, total_pages=9):
        self.product_id = product_id
        self.total_pages = total_pages
        self.base_url = f"https://www.amazon.com/{product_id}/product-reviews/{product_id}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
        self.urls = self.generate_review_urls()

    def generate_review_urls(self):
        return [f"{self.base_url}&pageNumber={i}" for i in range(1, self.total_pages + 1)]

    def scrape_reviews(self, url):
        params = {'api_key': "ce7562064ab1029af3f879b124f5233a", 'url': url}
        response = requests.get('http://api.scraperapi.com/', params=urlencode(params))
        soup = BeautifulSoup(response.text, 'html.parser')

        # Scrape star rating only from the first page
        product_star_rating = None
        if url == self.urls[0]:
            item = soup.find("span", {"data-hook": "rating-out-of-text"})
            product_star_rating = item.get_text()

        # Scraping review titles and contents
        titles = {i + 1: item.get_text().strip() for i, item in enumerate(soup.find_all("a", "review-title"))}
        reviews = [item.get_text() for item in soup.find_all("span", {"data-hook": "review-body"})]

        return titles, reviews, product_star_rating


# Class for managing the sentiment analysis process
class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    @staticmethod
    def preprocess_reviews(reviews):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        punctuation_free_reviews = [review.translate(str.maketrans('', '', string.punctuation)) for review in reviews]
        tokenized_reviews = [word_tokenize(review) for review in punctuation_free_reviews]
        cleaned_reviews = [[token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words] for
                           tokens in tokenized_reviews]
        return cleaned_reviews

    @staticmethod
    def load_adjective_vocab(file_path):
        with open(file_path) as file:
            return [line.strip() for line in file]

    @staticmethod
    def validate_reviews(reviews, vocab):
        return [[token for token in review if token in vocab] for review in reviews]

    def analyze_sentiment(self, reviews):
        sentiment_counts = {'very_positive': 0, 'positive': 0, 'negative': 0, 'very_negative': 0, 'neutral': 0}

        for review in reviews:
            if not review:
                continue
            score = sum(self.sia.polarity_scores(w)["compound"] for w in review) / len(review)
            if score > 0.05:
                sentiment_counts['positive'] += 1
            elif score > 0:
                sentiment_counts['very_positive'] += 1
            elif score < -0.05:
                sentiment_counts['negative'] += 1
            elif score < 0:
                sentiment_counts['very_negative'] += 1
            else:
                sentiment_counts['neutral'] += 1

        return sentiment_counts


# Class for reporting the results
class ReportGenerator:
    @staticmethod
    def display_results(total_sentiment_counts, product_star_rating):
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


# Main function to orchestrate the process
def main():
    nltk.download("all")

    product_id = "Apple-Generation-Cancelling-Personalized-Customizable/product-reviews/B0BDHWDR12"
    scraper = ReviewScraper(product_id)
    sentiment_analyzer = SentimentAnalyzer()
    total_sentiment_counts = {'very_positive': 0, 'positive': 0, 'negative': 0, 'very_negative': 0, 'neutral': 0}

    product = "Apple AirPods Pro (2nd Generation) Wireless Earbuds".upper()
    print("SENTIMENT ANALYSIS OF", product, "AMAZON'S REVIEWS\n")

    # Process each URL
    for i, url in enumerate(scraper.urls):
        print(f"\nPage {i + 1}:")
        titles, reviews, product_star_rating = scraper.scrape_reviews(url)
        print("\nTITLES:", titles)

        cleaned_reviews = sentiment_analyzer.preprocess_reviews(reviews)
        adj_vocab = sentiment_analyzer.load_adjective_vocab("wordnetAdj.txt")
        validated_reviews = sentiment_analyzer.validate_reviews(cleaned_reviews, adj_vocab)

        sentiment_counts = sentiment_analyzer.analyze_sentiment(validated_reviews)
        for key in total_sentiment_counts:
            total_sentiment_counts[key] += sentiment_counts[key]

    # Generate and display report
    ReportGenerator.display_results(total_sentiment_counts, product_star_rating)


# Run the main function
if __name__ == "__main__":
    main()


