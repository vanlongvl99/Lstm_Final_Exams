
from keras.models import load_model
from collections import Counter

model_test = load_model("./model4.h5")
model_test.load_weights('./model_weights4.h5')


def divide_phrases(text):
    phrase = r'\w[\w ]*|\s\W+|\W+'
    return re.findall(phrase, text)
 
def guess(ngram):
    text = ' '.join(ngram)
    prediction = model_test.predict(np.array([encode(text)]), verbose=0)
    return decode(prediction[0], calc_argmax=True).strip('\x00')
 
def add_accent_for_text(text):
    ngrams = list(gen_ngrams(text.lower(), n=NGRAM))
    guessed_ngrams = list(guess(ngram) for ngram in ngrams)
    candidates = [Counter() for _ in range(len(guessed_ngrams) + NGRAM - 1)]
    for nid, ngram in enumerate(guessed_ngrams):
        for wid, word in enumerate(re.split(' +', ngram)):
            candidates[nid + wid].update([word])
    output = ' '.join(c.most_common(1)[0][0] for c in candidates)
    return output

def accent_sentence(sentence):
  list_phrases = divide_phrases(sentence)
  output = ""
  for phrases in list_phrases:
    if len(phrases.split()) <= 2 or not re.match("\w[\w ]+", phrases):
      output += phrases
    else:
      output += add_accent(phrases)
      if phrases[-1] == " ":
        output += " "
  return output

text = 'toi di dao quanh thanh pho'
output = accent_sentence(text)
print(output)