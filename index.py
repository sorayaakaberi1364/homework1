import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# یک مجموعه متن ساده برای آموزش
corpus = [
    'این یک متن است',
    'ما در حال یادگیری ماشین هستیم',
    'این متن برای پیش‌بینی کلمات استفاده می‌شود',
    'مدل‌های یادگیری عمیق برای پردازش زبان طبیعی بسیار مفید هستند',
    'تجزیه و تحلیل داده‌ها بخش مهمی از یادگیری ماشینی است'
]

# ایجاد یک Tokenizer و یادگیری دیکشنری از متن‌ها
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

# تبدیل متن‌ها به توالی عددی
total_words = len(tokenizer.word_index) + 1  # +1 برای یک کلمه اضافی برای صفر
input_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# پیدا کردن طول حداکثر دنباله‌ها و ایجاد ورودی و خروجی
max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# ورودی‌ها و خروجی‌ها را تعیین کنید
X, y = input_sequences[:, :-1], input_sequences[:, -1]

# تبدیل y به دسته‌بندی
y = np.eye(total_words)[y]  # استفاده از one-hot encoding برای y
# ساخت مدل LSTM
model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_length - 1))
model.add(LSTM(128))
model.add(Dense(total_words, activation='softmax'))

# کامپایل مدل
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# آموزش مدل
model.fit(X, y, epochs=100, verbose=1)
def predict_next_word(model, tokenizer, input_text):
    sequence = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length - 1, padding='pre')
    predicted_word_index = model.predict(padded_sequence)
    return tokenizer.index_word[np.argmax(predicted_word_index[0])]  # پیش‌بینی کلمه بعدی

# مثال استفاده از تابع پیش‌بینی
input_text = "این یک متن"
predicted_word = predict_next_word(model, tokenizer, input_text)
print(f"کلمه بعدی پیش‌بینی شده: {predicted_word}")
