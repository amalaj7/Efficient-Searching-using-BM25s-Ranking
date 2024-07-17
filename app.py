import streamlit as st
import bm25s
import Stemmer 
import pandas as pd

# Setting the titles
st.set_page_config(page_title="Efficient Searching using BM25s Ranking",)
st.title("Efficient Searching using BM25s Ranking")

df = pd.read_csv("Mcdonalds_reviews.csv")
stemmer = Stemmer.Stemmer("english")

# Create the BM25 model 
retriever = bm25s.BM25()

def get_ranked_documents(query, review):
    # Tokenize the corpus and only keep the ids (faster and saves memory)
    corpus_tokens = bm25s.tokenize(review, stopwords="en", stemmer=stemmer)

    # index the corpus
    retriever.index(corpus_tokens)

    query_tokens = bm25s.tokenize(query, stemmer=stemmer)

    # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
    results, scores = retriever.retrieve(query_tokens, corpus= review, k= 5)
    for i in range(results.shape[1]):
        doc, score = results[0, i], scores[0, i]    
        if score >= 0.5:
            st.write(f"Rank {i+1} : {doc}")


city_name = st.selectbox("Select City in USA", options= list(set(df.city.to_list())), index = None)
city_df = df[df.city == city_name]
address = st.selectbox("Optional: Select Address", options= list(set(city_df.address.to_list())), index = None, placeholder= "Select address")

if address:
    shop_review = city_df[city_df.address == address].review.to_list()
else:
    shop_review = city_df.review.to_list()

query = st.text_input("Enter your Search Query: ", placeholder= "Search your query here ..")

if st.button("Predict"):
    get_ranked_documents(query, shop_review)