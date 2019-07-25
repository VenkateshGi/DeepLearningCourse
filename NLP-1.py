from tensorflow.keras.preprocessing.text import Tokenizer



sentences = ["Hello, How are you?","I am fine"]
token = Tokenizer(num_workds = 100)
token.fit_on_texts(sentences)
word_index = token.word_index(sentences)
print(word_index)

token.texts_to_sequences(sentences)
