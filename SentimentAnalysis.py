# -*- coding: utf-8 -*-

#Importing and downloading Natural Langauge Processing tool kits 
import nltk
nltk.download("all") #selects the entire set of book resources
from nltk import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

#Importing webscraping modules and libraries
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode

#Others
import string

Product = "Apple AirPods Pro (2nd Generation) Wireless Earbuds".upper()
print("SENTIMENT ANALYSIS OF", Product,"AMAZON'S REVIEWS\n")

#Number of urls depends on how many pages of review a product has so far
url1= "https://www.amazon.com/Apple-Generation-Cancelling-Personalized-Customizable/product-reviews/B0BDHWDR12/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
url2 = "https://www.amazon.com/Apple-Generation-Cancelling-Personalized-Customizable/product-reviews/B0BDHWDR12/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=2"
url3 = "https://www.amazon.com/Apple-Generation-Cancelling-Personalized-Customizable/product-reviews/B0BDHWDR12/ref=cm_cr_getr_d_paging_btm_next_3?ie=UTF8&reviewerType=all_reviews&pageNumber=3"
url4 = "https://www.amazon.com/Apple-Generation-Cancelling-Personalized-Customizable/product-reviews/B0BDHWDR12/ref=cm_cr_getr_d_paging_btm_next_4?ie=UTF8&reviewerType=all_reviews&pageNumber=4"
url5 = "https://www.amazon.com/Apple-Generation-Cancelling-Personalized-Customizable/product-reviews/B0BDHWDR12/ref=cm_cr_getr_d_paging_btm_next_5?ie=UTF8&reviewerType=all_reviews&pageNumber=5"
url6 = "https://www.amazon.com/Apple-Generation-Cancelling-Personalized-Customizable/product-reviews/B0BDHWDR12/ref=cm_cr_getr_d_paging_btm_next_6?ie=UTF8&reviewerType=all_reviews&pageNumber=6"
url7 = "https://www.amazon.com/Apple-Generation-Cancelling-Personalized-Customizable/product-reviews/B0BDHWDR12/ref=cm_cr_getr_d_paging_btm_next_7?ie=UTF8&reviewerType=all_reviews&pageNumber=7"
url8 = "https://www.amazon.com/Apple-Generation-Cancelling-Personalized-Customizable/product-reviews/B0BDHWDR12/ref=cm_cr_getr_d_paging_btm_next_8?ie=UTF8&reviewerType=all_reviews&pageNumber=8"
url9 = "https://www.amazon.com/Apple-Generation-Cancelling-Personalized-Customizable/product-reviews/B0BDHWDR12/ref=cm_cr_getr_d_paging_btm_next_9?ie=UTF8&reviewerType=all_reviews&pageNumber=9"

URLs = [url1,url2,url3,url4,url5,url6,url7,url8,url9]

Total_VeryPositive_Reviews = 0
Total_Positive_Reviews = 0
Total_Negative_Reviews = 0
Total_VeryNegative_Reviews = 0
Total_Neutral_Reviews = 0

for i,url in enumerate(URLs):
    print("\npage ", i+1,":") #Keeps track of page numbers
    params = {'api_key': "ce7562064ab1029af3f879b124f5233a", 'url':url }
    response = requests.get('http://api.scraperapi.com/',params=urlencode(params))
    soup = BeautifulSoup(response.text, 'html.parser')
    if i == 0:
        #Scraping the star rating part
        item = soup.find("span",{"data-hook":"rating-out-of-text"})
        product_star_rating = item.get_text()
    data_string = ""
    #Scraping the titles for each page of reviews/url
    titles = dict()
    for review_number,item in enumerate(soup.find_all("a", "review-title")):
          data_string = data_string + item.get_text()
          if (review_number+1) not in titles:
              titles[review_number+1] = data_string.strip()
              data_string = ""
          data_string = ""
    print("\nTITLES: ",titles)
    
    #Scraping review content contained in each page corresponding to the titles above
    reviews = []   
    for item in soup.find_all("span", {"data-hook": "review-body"}):
          data_string = data_string + item.get_text()
          reviews.append(data_string)
          data_string= ""
   

##Preprocessing and cleaning each review as follows:
    
    #Removing punctuation
    punctuationfree_reviews =[review.translate(str.maketrans('','',string.punctuation)) for review in reviews]
    
    #Creating a list of tokens from the review
    tokenized_reviews =[word_tokenize(review) for review in punctuationfree_reviews]
    
    
    #Removing numerals and special characters from the tokens list as well as common stopwords
    stopWords = set(nltk.corpus.stopwords.words('english'))
    alphalowercase_reviews =[]
    for review_tokens in tokenized_reviews:
        alreview = []
        for i in range(len(review_tokens)):
            if review_tokens[i].isalpha() and review_tokens[i] not in stopWords:
                r = review_tokens[i].lower()
                alreview.append(r)
        alphalowercase_reviews.append(alreview)
                
         

##building a vocabulary of acceptable ADJECTIVES found in WordNet
    vocab = []
    with open("wordnetAdj.txt") as WordNetinputfile:
         for line in WordNetinputfile:
            newTerm = line.split()
            vocab.append(newTerm[0])
    WordNetinputfile.close()
    
    
#building a vocabulary of acceptable ADVERBS found in WordNet
    # with open("wordnetAdv.txt") as WordNetinputfile:
    #      for line in WordNetinputfile:
    #         newTerm = line.split()
    #         vocab.append(newTerm[0])
    # WordNetinputfile.close()


    
#Limiting our Reviews to just valid words (defined by WordNet)
    validated_reviews = []
    for review in alphalowercase_reviews:
        valid_review = []
        
        for token in review:
            if token in vocab:
                
                valid_review.append(token)
        validated_reviews.append(valid_review)
    
       
##Sentiment Analysis of Reviews using sia.polarity scores
    
    sia = SentimentIntensityAnalyzer()
    reviews_sentiment = []
    
    Vpos_reviews_perpage = 0
    Pos_reviews_perpage = 0
    Neg_reviews_perpage = 0
    Vneg_reviews_perpage = 0
    Neu_reviews_perpage = 0
    
    for review_number, review in enumerate(validated_reviews):
        
        i=0
        for w in review:
           w_sentiment = sia.polarity_scores(str(w))["compound"]
           i+= w_sentiment
           
        review_sentiment_score =round(i/len(review),4)
        k = review_sentiment_score
        
        if 0<k<=1:
            print("Sentiment score = ",k, ", Review number",review_number+1,"is positive ")
            Pos_reviews_perpage+= 1
            
        elif k>1:
            print("Sentiment score = ",k, ", Review number",review_number+1,"is very positive ")
            Vpos_reviews_perpage+= 1
            
        elif -1<=k<0:
            print("Sentiment score = ",k, ", Review number",review_number+1,"is negative")
            Neg_reviews_perpage+= 1
            
        elif k<-1:
            print("Sentiment score = ",k, ", Review number",review_number+1,"is very negative")
            Vneg_reviews_perpage+= 1
            
        else:
            print("Sentiment score = ",k, ", Review number",review_number+1,"is neutral")
            Neu_reviews_perpage+= 1
            
        
      
    Total_VeryPositive_Reviews+= Vpos_reviews_perpage
    Total_Positive_Reviews+= Pos_reviews_perpage
    Total_Negative_Reviews+= Neg_reviews_perpage
    Total_VeryNegative_Reviews+= Vneg_reviews_perpage
    Total_Neutral_Reviews+= Neu_reviews_perpage   
Total_Reviews = Total_VeryPositive_Reviews+Total_Positive_Reviews+Total_Negative_Reviews+Total_VeryNegative_Reviews+ Total_Neutral_Reviews

Overall_Positive_Reviews = "{:.4%}".format((Total_VeryPositive_Reviews+Total_Positive_Reviews)/Total_Reviews)
Overall_Negative_Reviews = "{:.4%}".format((Total_VeryNegative_Reviews+Total_Negative_Reviews)/Total_Reviews)
Overall_Neutral_Reviews = "{:.4%}".format((Total_Neutral_Reviews)/Total_Reviews)

print("\n")
print(f"Total Reviews = {Total_Reviews}")
print(f"Total Very Positive Reviews = {Total_VeryPositive_Reviews}\tTotal Positive Reviews = {Total_Positive_Reviews}")
print(f"Total Very Negative Reviews = {Total_VeryNegative_Reviews}\tTotal Negative Reviews = {Total_Negative_Reviews}")
print(f"Total Neutral Reviews = {Total_Neutral_Reviews}")
print("\nOverall Positive Reviews: ",Overall_Positive_Reviews)      
print("Overall Negative Reviews: ", Overall_Negative_Reviews)     
print("Overall Neutral Reviews: ", Overall_Neutral_Reviews)  
print("\n Compare to the product's star rating of ", product_star_rating)
    


    
