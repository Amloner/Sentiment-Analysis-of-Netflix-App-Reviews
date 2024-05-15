#Dictionary-based Sentiment Analyzer
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize,TreebankWordTokenizer
from nltk.corpus import stopwords
import regex as re
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import word_tokenize, pos_tag
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix

tb_tokenizer = TreebankWordTokenizer()

reviews = pd.read_csv("/Users/aaudit/Documents/Projects/Netflix_review/Dataset_exploration/small_corpus.csv")


def text_clean(text, method, rm_stop):
    """
        Cleans text and reduces words to their base forms 
    """
    text = re.sub(r"\n","",text)   #remove line breaks
    text = text.lower() #convert to lowercase
    text = re.sub(r"\d+","",text)   #remove digits and currencies 
    text = re.sub(r'[\$\d+\d+\$]', "", text)
    text = re.sub(r'\d+[\.\/-]\d+[\.\/-]\d+', '', text)   #remove dates 
    text = re.sub(r'\d+[\.\/-]\d+[\.\/-]\d+', '', text)
    text = re.sub(r'\d+[\.\/-]\d+[\.\/-]\d+', '', text)
    text = re.sub(r'[^\x00-\x7f]',r' ',text)   #remove non-ascii
    text = re.sub(r'[^\w\s]','',text)   #remove punctuation
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)   #remove hyperlinks

    #remove stop words
    if rm_stop == True:
        filtered_tokens = [word for word in word_tokenize(text) if not word in set(stopwords.words('english'))]
        text = " ".join(filtered_tokens)

    #lemmatization: typically preferred over stemming
    if method == 'L':
        lemmer = WordNetLemmatizer()
        lemm_tokens = [lemmer.lemmatize(word) for word in word_tokenize(text)]
        return " ".join(lemm_tokens)

    #stemming
    if method == 'S':
        porter = PorterStemmer()
        stem_tokens = [porter.stem(word) for word in word_tokenize(text)]
        return " ".join(stem_tokens)

    return text

def penn_to_wn(tag):
    """
        Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


#Stop words are not removed as they are critical for sentiment anaylsis
reviews["content_lower"] = reviews['content'].apply(lambda rev: text_clean(rev, 'L', False))

#Words are tokenized
reviews["tb_token"] = reviews['content_lower'].apply(lambda rev: tb_tokenizer.tokenize(str(rev)))



def get_sentiment_score(tokens):
    score = 0
    tags = pos_tag(tokens)
    for word, tag in tags:
        wn_tag = penn_to_wn(tag)
        if not wn_tag:
            continue
        synsets = wn.synsets(word, pos=wn_tag)
        if not synsets:
            continue
        
        #most common set:
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        
        score += (swn_synset.pos_score() - swn_synset.neg_score())
        
    return score
                    

reviews['sentiment_score'] = reviews['tb_token'].apply(lambda tokens: get_sentiment_score(tokens))

#Sentiment score is converted into categorical values
reviews['sentiment_score'] = reviews['sentiment_score'].apply(lambda x: "positive" if x>1 else ("negative" if x<0.5 else "neutral"))

#The actual rating of the reviews into categorical values
reviews['true_sentiment'] = reviews['score'].apply(lambda x: "positive" if x>=4 else ("neutral" if x==3 else "negative"))


y_swn_pred, y_true = reviews['sentiment_score'].tolist(), reviews['true_sentiment'].tolist()

#The precision and recall of the sentiment anaylser is compared to the actual ratings


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