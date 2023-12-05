from django.shortcuts import render
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import requests
from subprocess import run, PIPE
inp=''
tweet=''


    # return HttpResponse("Success")
# tweet = "@MehranShakarami today's cold @ home 😒 https://mehranshakarami.com"
def button(request):
    return render(request,'home.html',{})
def external(request):
    global inp
    global tweet
    inp=request.POST.get('param')
    tweet=inp
    inp=''
    return output(request)
    # return HttpResponse("Success")
def output(request):
    global tweet
    # tweet = " Marathi is a language. 'Aho aikana?' Anyone would fall for it."
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    tweet_proc = " ".join(tweet_words)
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores=scores.tolist()
    p=list(zip(labels,scores))
    for i in range(len(scores)):
        
        l = labels[i]
        s = scores[i]
        print(l,s)
    tweet=''
    
    return render(request,'home.html',{'data':p})

    