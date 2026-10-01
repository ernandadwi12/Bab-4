# EXERCISE 4.19
# Sentiment Analysis menggunakan Transformers

from transformers import pipeline

# Membuat classifier sentiment analysis
classifier = pipeline('sentiment-analysis')

# Kalimat positif dan negatif
sentences = [
    "I really love this product.",
    "The movie was amazing and very enjoyable.",
    "The service was excellent.",
    "I am very happy with the results.",
    
    "I hate this product.",
    "The movie was terrible and boring.",
    "The service was very bad.",
    "I am disappointed with the results."
]

# Melakukan analisis setiap kalimat
for sentence in sentences:
    result = classifier(sentence)

    print("Sentence :", sentence)
    print("Result   :", result[0]["label"])
    print("Score    :", round(result[0]["score"], 4))
    print("-" * 50)