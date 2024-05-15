import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data= pd.read_csv("/Users/aaudit/Documents/Projects/Netflix_review/Dataset_creation/netflix_reviews.csv")

one_1500 = data[data['score']==1.0].sample(n=1500)
two_500 = data[data['score']==2.0].sample(n=500)
three_500 = data[data['score']==3.0].sample(n=500)
four_500 = data[data['score']==4.0].sample(n=500)
five_1500 = data[data['score']==5.0].sample(n=1500)

undersampled_dataset = pd.concat([one_1500,two_500,three_500,four_500,five_1500] , axis=0)

random_dataset = data.sample(n=100000, random_state=31)

undersampled_dataset.to_csv("/small_corpus.csv", index=False)

random_dataset.to_csv("/big_corpus.csv", index=False)
