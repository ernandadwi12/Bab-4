# Exercise 4.20: teks dan pertanyaan berbeda dari Example 4.24
from transformers import pipeline

question_answerer = pipeline('question-answering')

context = ("Surabaya is the capital of East Java province in Indonesia. "
           "It is the second largest city in the country after Jakarta. "
           "The city was founded in 1293 and is known as the City of Heroes "
           "because of the Battle of Surabaya in November 1945. "
           "The population is about 3 million people.")

questions = [
    'What is the capital of East Java?',
    'Which city is the largest in Indonesia?',          # jawaban tidak tertulis langsung (butuh inferensi)
    'When was the city founded?',
    'Why is Surabaya known as the City of Heroes?',
    'How many people live in Surabaya?',
    'Who is the mayor of Surabaya?',                    # jawaban TIDAK ada di teks
]

for q in questions:
    result = question_answerer(question=q, context=context)
    print("Q     :", q)
    print("Answer:", result['answer'])
    print("Score : {:.4f}  (start={}, end={})".format(
        result['score'], result['start'], result['end']))
    print()