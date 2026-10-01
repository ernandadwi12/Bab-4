# EXERCISE 4.18
# Analisis teks menggunakan NLTK

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download resource NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')

# Teks baru yang akan dianalisis
text = """
Artificial intelligence is changing the way people learn and work.
Artificial intelligence can help students understand difficult topics.
Students can use technology to learn faster and improve their skills.
"""

# Mengubah teks menjadi huruf kecil
text_lower = text.lower()

# Tokenisasi
tokens = word_tokenize(text_lower)

# Mengambil kata saja
words = [
    word for word in tokens
    if word.isalpha()
]

# Menghilangkan stopwords
stop_words = set(stopwords.words('english'))

filtered_words = [
    word for word in words
    if word not in stop_words
]

# Menghitung frekuensi kata
word_frequency = Counter(filtered_words)

# Menampilkan hasil
print("=== TEXT ANALYSIS ===")

print("\nTeks:")
print(text)

print("\nJumlah kata sebelum stopword:")
print(len(words))

print("\nJumlah kata setelah stopword:")
print(len(filtered_words))

print("\nKata setelah stopword dihilangkan:")
print(filtered_words)

print("\nFrekuensi kata:")
for word, count in word_frequency.most_common():
    print(word, ":", count)