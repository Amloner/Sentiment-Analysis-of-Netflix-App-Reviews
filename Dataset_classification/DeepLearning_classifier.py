#Deep-Learning-based Sentiment Analyzer
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix


from transformers import pipeline
sentiment_pipeline = pipeline('sentiment-analysis', model="finiteautomata/bertweet-base-sentiment-analysis")


reviews = pd.read_csv("/Users/aaudit/Documents/Projects/Netflix_review/Dataset_exploration/small_corpus.csv")

#Long reviews were truncated to 200 characters due to model limitations
reviews['sentiment_score'] = reviews['content'].apply(lambda x: sentiment_pipeline(x[:200])[0]['label'])

#Sentiment analysis result is expanded
reviews['sentiment_score'] = reviews['sentiment_score'].apply(lambda x: "positive" if x=='POS' else ("negative" if x=='NEG' else "neutral"))

#The actual rating of the reviews into categorical values
reviews['true_sentiment'] = reviews['score'].apply(lambda x: "POS" if x>=4 else ("neutral" if x==3 else "negative"))


y_swn_pred, y_true = reviews['sentiment_score'].tolist(), reviews['true_sentiment'].tolist()

def evaluvate_model(y_true, y_swn_pred,reviews):
    sns.countplot(x='score', hue='sentiment_score' ,data = reviews,palette=("tab10"))
    plt.savefig('countPlot.png')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (12,7))
    sns.boxenplot(x='score', y='sentiment_score', data = reviews, ax=ax,palette=("tab10"))
    plt.savefig('boxPlot.png')
    cm = confusion_matrix(y_true, y_swn_pred)
    fig , ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    sns.heatmap(cm, cmap='viridis_r', annot=True, fmt='d', square=True, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.savefig('confusionMatrix.png')
    clsf_report = pd.DataFrame(classification_report(y_true = y_true, y_pred = y_swn_pred, output_dict=True)).transpose()
    clsf_report.to_csv('classificationreport.csv', index= True)


evaluvate_model(y_true, y_swn_pred,reviews)
