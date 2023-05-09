from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import re

class SentimentLabels():

  def __init__(self):
    
    task='sentiment'
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    self.tokenizer = AutoTokenizer.from_pretrained("./model")

    # download label mapping
    self.labels=[]
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    self.labels = [row[1] for row in csvreader if len(row) > 1]
    self.model = AutoModelForSequenceClassification.from_pretrained("./model")
  
  def preprocess(self,text):
    new_text = []
 
 
    for t in text.split(" "):
        t = re.sub(r'#\S+', '', t)
        t = re.sub(r'@\S+', '', t)
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

  

  def get_sentiment(self,text):
    try:
      text = self.preprocess(text)
      encoded_input = self.tokenizer(text, return_tensors='pt')
      output = self.model(**encoded_input)
      scores = output[0][0].detach().numpy()
      scores = softmax(scores)
      ranking = np.argsort(scores)
      ranking = ranking[::-1]
      sentiment = []
      for i in range(scores.shape[0]):
          l = self.labels[ranking[i]]
          sentiment.append(l)
          s = scores[ranking[i]]
      return sentiment[0]
    except Exception as e:
      print("can't able to predict sentiment. The problem is",e)
