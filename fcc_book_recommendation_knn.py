#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries (you may add additional imports but you may not have to)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


# In[2]:


# get data files
get_ipython().system('wget -nc -q https://cdn.freecodecamp.org/project-data/books/book-crossings.zip')

get_ipython().system('unzip -oq book-crossings.zip')

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'


# In[3]:


# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})


# In[4]:


# Keep only statistically significant data:
# users with at least 200 ratings and books with at least 100 ratings
user_counts = df_ratings['user'].value_counts()
book_counts = df_ratings['isbn'].value_counts()

df = df_ratings[
    df_ratings['user'].isin(user_counts[user_counts >= 200].index) &
    df_ratings['isbn'].isin(book_counts[book_counts >= 100].index)
]

# Attach titles and remove duplicate (title, user) pairs (same book, several ISBNs)
df = df.merge(df_books, on='isbn').drop_duplicates(['title', 'user'])

# Book x user rating matrix (0 = not rated)
book_user = df.pivot(index='title', columns='user', values='rating').fillna(0)
print(book_user.shape)

# KNN model using cosine distance
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(csr_matrix(book_user.values))


# In[5]:


# function to return recommended books - this will be tested
def get_recommends(book = ""):
  distances, indices = model.kneighbors(book_user.loc[[book]].values, n_neighbors=6)
  # skip the first neighbour (the book itself); list from farthest to closest
  recs = [[book_user.index[i], float(d)] for i, d in zip(indices[0][1:], distances[0][1:])][::-1]
  recommended_books = [book, recs]
  return recommended_books

get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")


# In[6]:


books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2): 
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You haven't passed yet. Keep trying!")

test_book_recommendation()

