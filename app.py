# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:18:24 2023

@author: iT LOGIX
"""
from fastapi import FastAPI
import requests
#from mangum import Mangum
import uvicorn
from sentiment import SentimentLabels

sentiment_obj = SentimentLabels()
print("successfully loaded Ner class")

app = FastAPI()

    
#handler = Mangum(app)

@app.get("/")
def read_root():
    return {"hello":"World"}

@app.get("/getsentiment")
async def getsentiment(text:str):
    try:
        sentiment_label = sentiment_obj.get_sentiment(text)
        return {"Sentiment":sentiment_label}
    except Exception as e:
        print("something went wrong while predicting sentiment",e)
        return {"Sentiment":''}
    

if __name__ =="__main__":
    uvicorn.run(app, host="0.0.0.0",port=9000)