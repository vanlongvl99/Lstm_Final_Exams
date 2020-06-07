!pip install unidecode

 
import re
import unidecode
import itertools
from nltk import ngrams
import string
import numpy as np
from tqdm import tqdm
import os


from google.colab import drive
drive.mount('/content/gdrive')


#load data
# link load data https://drive.google.com/drive/folders/19mfnUc1p5Dl0cQnONsPWYrulAtFJXtid?usp=sharing
with open("./train_data.txt", "r") as f_r:
    total_lines = f_r.read().split("\n")
    
print(len(total_lines))
for line in total_lines[:5]:
  print(line)


# Dựa vào 6 từ trước đó, và mỗi từ có tối đa 6 ký tự (trừ từ nghiêng) => 6*6=36
max_len = 36
NGRAM = 6
BATCH_SIZE = 256

# def hàm xóa dấu cho từ
# input text: là 1 đoạn text có dấu
# output: là 1 đoạn text không có dấu
def remove_accent(text):
    return unidecode.unidecode(text)
text_ex = "hôm nay, tôi đi học ở trường bách khoa (BK)"
print(remove_accent(text_ex))


# Tách 1 câu thành các đoạn nhỏ (phrases) bởi dấu câu trong câu
# input text: là 1 đoạn text
# output: là 1 list chứa các đoạn nhỏ được tách bởi dấu ',' trong text
def extract_phrases(text):
    return re.findall(r'\w[\w ]+', text)
print(extract_phrases(text_ex))


#Tiến hành tách phrases, các phrases có ít hơn 2 token sẽ bị loại bỏ
phrases = itertools.chain.from_iterable(extract_phrases(text) for text in total_lines[:10000])
phrases = [p.strip() for p in phrases if len(p.split()) > 1]

print(len(phrases))
print(total_lines[0])
print(phrases[:20])



#định nghĩa lại bảng chữ cái, alphabet chính là đầu ra mong muốn tại mỗi step của mô hình.

accented_chars_vietnamese = [
    'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
    'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ',
    'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ',
    'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự',
    'í', 'ì', 'ỉ', 'ĩ', 'ị',
    'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ',
    'đ',
]
accented_chars_vietnamese.extend([c.upper() for c in accented_chars_vietnamese])
alphabet = list(('\x00 _' + string.ascii_letters + string.digits + ''.join(accented_chars_vietnamese)))

print(len(alphabet))
print(alphabet)



# Sinh ra 6-grams bằng thư viện nltk
def gen_ngrams(words, n=6):
    return ngrams(words.split(), n)
    
list_ngrams = []
 
for p in tqdm(phrases):
  for ngr in gen_ngrams(p, NGRAM):
    if len(" ".join(ngr)) < 36:
      list_ngrams.append(" ".join(ngr))

# del phrases
list_ngrams = list(set(list_ngrams))
print("")
print("len list ngrams:",len(list_ngrams))
print(list_ngrams[:10])


# chuyển text thành one-hot-vector, 2 hàm encode và decode
def encode(text, maxlen=max_len):
        text = "\x00" + text
        x = np.zeros((maxlen, len(alphabet)))
        for i, c in enumerate(text[:maxlen]):
            x[i, alphabet.index(c)] = 1
        if i < maxlen - 1:
          for j in range(i+1, maxlen):
            x[j, 0] = 1
        return x
 
def decode(x, calc_argmax=True):
    if calc_argmax:
        x = x.argmax(axis=-1)
    return ''.join(alphabet[i] for i in x)

# model

from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, LSTM, Bidirectional
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
 
HIDDEN_SIZE = 256
 
model = Sequential()

model.add(LSTM(HIDDEN_SIZE, input_shape=(max_len, len(alphabet)), return_sequences=True))

model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout=0.3, recurrent_dropout=0.15)))

model.add(TimeDistributed(Dense(len(alphabet))))

model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
 
model.summary()
# Chia data thành 2 tập train và test

from sklearn.model_selection import train_test_split
 
train_data, valid_data = train_test_split(list_ngrams, test_size=0.2, random_state=2019)
print(len(train_data))
print("train data:",train_data[:10])
print(len(valid_data))
#Vì data khá nhiều nên phải chia ra

def generate_data(data, batch_size=128):
    cur_index = 0
    while True:
        
        x, y = [], []
        for i in range(batch_size):  
            y.append(encode(data[cur_index]))
            x.append(encode(unidecode.unidecode(data[cur_index])))
            cur_index += 1
            
            if cur_index > len(data)-1:
                cur_index = 0
        
        yield np.array(x), np.array(y)

train_generator = generate_data(train_data, batch_size=BATCH_SIZE)
validation_generator = generate_data(valid_data, batch_size=BATCH_SIZE)
 

  
model.fit_generator(train_generator, steps_per_epoch=len(train_data)//BATCH_SIZE, epochs=3,
                    validation_data=validation_generator, validation_steps=len(valid_data)//BATCH_SIZE)


model.save('model4.h5')
model.save_weights('model_weights4.h5')


