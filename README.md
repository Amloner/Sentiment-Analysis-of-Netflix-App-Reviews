# Sentiment Analysis of Netflix App Reviews
This project aims to analyse reviews of the Netflix application using NLP techniques. A rudimentary dictionary based model and an advanced deep learning model will be used to analyse and the performance of these models will be evaluvated.

### Techniques Employed
   * Sampling from imbalanced datasets using the imbalanced-learn package
   * Enquiring about the sentiment value of the reviews with the dictionary-based sentiment analysis tools from NLTK, a natural language processing toolkit, used in Python.
   * Analyzing the reviews with a state-of-the-art deep learning technique, namely with the Bertweet model.
   * Evaluating your model and creating descriptive statistics in Python with scikit-learn library.
  
   
### Project Outline

The project consists of four steps
   * Creating the dataset.
   * Creating a dictionary-based sentiment analyzer.
   * Creating neural network-based sentiment analyzers.
   * Evaluating both the sentiment analyzers.
     
### Dataset

The Netlifx review dataset can be downloaded from [here](https://www.kaggle.com/datasets/ashishkumarak/netflix-reviews-playstore-daily-updated?resource=download).



## Creating Dataset
   * Read the netflix app review data.
   * Create a plot of the ratings of the product. Study the distribution of the categories.
   * Take a random sample of the reviews by selecting 1500 reviews with rating 1, 500-500-500 reviews with ratings 2, 3, 4, and 1500 reviews with rating 5. This gives you a smaller balanced corpus.
   * Take a random sample of the reviews by selecting 100,000 reviews, this gives you a bigger representative corpus.
   * Export your corpora to two separate .csv files. Both of your tables should contain a column for the reviews and a column for the ratings.

## Creating dictionary-based sentiment analyzer
   * Read the smaller balanced corpus.
   * Clean the data - remove linebreaks, digits, dates, etc.
   * Perform leemmatization of the reviews
   * Tokenize using TreebankWordTokenizer
   * Perform sentiment analysis using wordnet

## Creating neural network-based sentiment analyzer
   * Read the smaller balanced corpus.
   * Truncate longer reviews.
   * Import pipeline from transformers library
   * Download bertweet-sentiment-analysis model. [Model was chosen as it categorised text as POS, NEG, and NEU unlike most models which are binary]
   * Analyse sentiment using the model

## Evaluating both the sentiment analyzers.
   * Visualise the performance of the models using confusion matrix and box plots.
   * Use classification report from sklearn to evaluvate how the models have performed
