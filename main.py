from flask import Flask
from flask import render_template,request
from flask_wtf import FlaskForm
from wtforms import widgets, StringField, SubmitField, TextAreaField, IntegerField, validators
import numpy as np
import nltk
nltk.download('punkt')

from nltk import word_tokenize,sent_tokenize
import string
app = Flask(__name__)
app.config.from_object("config.ProductionConfig")

class TextForm(FlaskForm):
    text = TextAreaField('text',validators=[
        validators.DataRequired()
    ])
    num = IntegerField('num', widget=widgets.Input(input_type='number'),validators=[
        validators.DataRequired()
    ])
    submit = SubmitField('summarize')

@app.route('/', methods=['GET','POST'])
def home():
    form = TextForm()
    if form.is_submitted():
        result = request.form
        num = int(result.get('num'))
        text = result.get('text')
        return render_template('summary.html', result=convert(num, text))
    #output,t2 = convert(5, "In Asia. In Asia, there are more than cases reported worldwide. As of Sunday, there were more than 58.2 million reported cases of COVID-19 worldwide, with more than 37.2 million of those cases listed as recovered, according to a COVID-19 tracking tool maintained by Johns Hopkins University. The global death toll stood at more than 1.3 million. In Asia, the daily tally of reported cases in Japan hit a record for the fourth day in a row, with 2,508 people confirmed infected, the Health Ministry said Sunday. A flurry of criticism has erupted, from opposition legislators and the public, slamming the government as having acted too slowly in halting its \"GoTo\" campaign, which encouraged travel and dining out with discounts. In Europe, French authorities ordered the culling of all minks at a farm after analysis showed a mutated version of the coronavirus was circulating among the animals. The move follows virus developments in mink farms in Denmark and other countries, including the Netherlands, Sweden and Greece. In the Americas, Chile says it will open its main border crossing and principal airport to foreign visitors on Monday after an eight-month pandemic shutdown. Arrivals will have to present evidence of a recent negative test for the novel coronavirus, as well as health insurance. They'll also have to report their whereabouts and health status for a two-week watch period. Those coming from high-risk countries will have to quarantine for 14 days. In Africa, Sudan's minister of cabinet affairs on Sunday tested positive for the coronavirus, the prime minister's office said, the latest in a string of senior officials to be infected as the country shows an increase of confirmed cases of COVID-19. Over the past month, acting ministers of finance and health, the central bank governor and two associates to Prime Minister Abdalla Hamdok have tested positive.")
    return render_template('index.html', form=form)

def convert(n, text):
    unique = list(set(word_tokenize(text.lower().translate(str.maketrans('', '', string.punctuation)))))
    unique.sort()
    sentences = sent_tokenize(text)
    if len(sentences)<n:
        return "Number of sentences requested was more than number of sentences in original text."
    final =[]
    matrix=[]
    for sentence in sentences:
        final.append(word_tokenize(sentence.lower().translate(str.maketrans('', '', string.punctuation))))


    
    for sentence in final:
        vector=[]
        for word in unique:
            count=sentence.count(word)
            vector.append(count)
        matrix.append(vector)
    
    matrix = np.asarray(matrix).transpose()

    u,d,v = np.linalg.svd(matrix)

    sen=[]

    for num in range(0,n):
        i=np.argmax(v[num])
        sen.append(i)

    summary = list(set(sen))
    summary.sort()

    final=""

    #for k in summary:
    #   final+=sentences[k] + " "


    #modification

    mod = np.dot(np.absolute(np.asarray(d)), np.absolute(np.asarray(v)))

    temp = mod.copy()

    ordersentences = []
    
    for p in range(0,n):
        ordersentences.append(np.argmax(mod)) 
        mod[np.argmax(mod)]=min(mod)

    ordersentences.sort()

    for index in ordersentences:
        final+=sentences[index]+" "

    return final
    #return [temp, np.dot(np.absolute(np.asarray(d)),np.absolute(np.transpose(np.asarray(v)))[0]),np.absolute(np.transpose(np.asarray(v)))[4]]