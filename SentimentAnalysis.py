# -*- coding: utf-8 -*-

# Importing necessary modules
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet
import string

# Class for managing the review scraping process
class ReviewScraper:
    def __init__(self, product,product_id, scraper_api_key):
        self.product = product
        self.product_id = product_id
        self.scraper_api_key = scraper_api_key
        self.base_url =  self.base_url = f"https://www.amazon.com/{product}/product-reviews/{product_id}/ref=cm_cr_getr_d_paging_btm_prev_1?ie=UTF8&reviewerType=all_reviews&pageNumber=1"

    def get_soup(self, url):
        # Request the page using ScraperAPI
        params = {
            'api_key': self.scraper_api_key,
            'url': url,
            'keep_headers': 'true'
        }
        response = requests.get(f'http://api.scraperapi.com/', params=urlencode(params))

        # Check if the request was successful
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
        else:
            print(f"Error fetching page: {response.status_code}")
            #print(f"Response: {response.text}")  # Print response for debugging
            return None

    def scrape_reviews(self):
        reviews = []
        titles = {}
        page_num = 1
        current_url = self.base_url
        product_star_rating = None
        j = 0
        while current_url:
            print(f"Scraping page {page_num}...")  # To track progress

            soup = self.get_soup(current_url)
            #print(soup.prettify())

            if soup is None:
                print(f"Failed to scrape page {page_num}")
                break

            # Scrape product star rating (only from the first page)
            if page_num == 1:
                product_star_rating_element = soup.find("span", {"data-hook": "rating-out-of-text"})
                if product_star_rating_element:
                    product_star_rating = product_star_rating_element.get_text().strip()
                else:
                    print("Star rating not found.")

            # Scrape review titles and contents
            review_titles = {i + 1 + j: item.get_text().strip() for i, item in
                             enumerate(soup.find_all("a", "review-title"))}
            review_contents = [item.get_text().strip() for item in soup.find_all("span", {"data-hook": "review-body"})]

            titles.update(review_titles)
            j += 1
            print(len(titles))
            reviews.extend(review_contents)
            print(len(reviews))

            # Check for the next page and update the URL
            next_page_element = soup.find("link", rel="next")
            if next_page_element and 'href' in next_page_element.attrs:
                next_page_url = next_page_element['href']
                # Construct the full URL for the next page
                current_url = next_page_url
                page_num += 1
            else:
                print("No more pages found.")
                current_url = None  # Exit the loop if there are no more pages

        return titles, reviews, product_star_rating

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()


    @staticmethod
    def preprocess_reviews(reviews):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        punctuation_free_reviews = [review.translate(str.maketrans('', '', string.punctuation)) for review in reviews]
        tokenized_reviews = [word_tokenize(review) for review in punctuation_free_reviews]

        # Applying lemmatization in the preprocessing step
        lemmatizer = WordNetLemmatizer()
        cleaned_reviews = [[lemmatizer.lemmatize(token.lower()) for token in tokens if
                               token.isalpha() and token.lower() not in stop_words]
                              for tokens in tokenized_reviews]
        return cleaned_reviews

    @staticmethod
    def load_adjective_vocab():
        # Extract adjectives ('a') from WordNet
        adjectives = set(wordnet.all_lemma_names(pos='a'))
        return list(adjectives)

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

#Class for reporting the results
class ReportGenerator:
    @staticmethod
    def display_results(total_sentiment_counts, product_star_rating,titles):
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
        print("\nCompare to the product's star rating of", product_star_rating, "and review titles below:\n")
        print(titles)






# Example usage
def main():
    #nltk.download("all")

    #Testing the code on a product from Amazon

    
    product = input("Enter the product name (e.g., 'Apple-Generation-Cancelling-Transparency-Personalized'): ")
    product_id = input("Enter the product ID (e.g., 'B0CHWRXH8B'): ")
    scraper_api_key = input("Enter your ScraperAPI key: ")

    scraper = ReviewScraper(product,product_id, scraper_api_key)
    titles, reviews, product_star_rating = scraper.scrape_reviews()

    print(f"SENTIMENT ANALYSIS OF {product.upper()} AMAZON'S REVIEWS\n")
    #print(reviews)


    sentiment_analyzer = SentimentAnalyzer()

    # Preprocess the reviews (tokenization, lemmatization, stopword removal)
    cleaned_reviews = sentiment_analyzer.preprocess_reviews(reviews)

    # Load adjective vocabulary for sentiment validation
    adj_vocab = sentiment_analyzer.load_adjective_vocab()

    # Validate reviews against the adjective vocabulary
    validated_reviews = sentiment_analyzer.validate_reviews(cleaned_reviews, adj_vocab)

    # Analyze sentiment of the reviews
    sentiment_counts = sentiment_analyzer.analyze_sentiment(validated_reviews)

    reportGenerator = ReportGenerator()

    reportGenerator.display_results(sentiment_counts, product_star_rating,titles)




if __name__ == "__main__":
    main()

