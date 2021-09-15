# Cojocariu Sebastian - Group 407 IA

import nltk
import urllib
import numpy as np
from num2words import num2words
from word2number import w2n
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.collocations import *
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.corpus import stopwords
import string 


# ex1 - download book
url = "https://www.gutenberg.org/files/64608/64608-0.txt"
content = urllib.request.urlopen(url).read().decode('utf-8')
n = len(content)


# ex2 - remove header and retain only the relevant part of the text
start_idx, end_idx = None, None
first_sentece = 'Demaris opened the gate and walked up the narrow path. There was a low'
end_sentence = 'break in her voice, “let’s go out an’ eat our Christmas dinner.'
for i in range(n):
	if content[i: i + len(first_sentece)] == first_sentece:
		start_idx = i
		break

for i in range(n - 1, -1, -1):
	if content[i - len(end_sentence) + 1: i + 1] == end_sentence:
		end_idx = i
		break

# get only the relevant part of the text and remove the rest
content = content[start_idx: end_idx + 1]
# remove traiiling spaces, tabs, newlines, etc.
content = " ".join(content.split())

# ex3 -  average_length + the number of sentences
# split into sentences
sentences = sent_tokenize(content)
print("\nNumber of sentences in the text: {}".format(len(sentences)))

# calculate lengths (include punctuation as words) for each sentence
sentences_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
# or if we dont want to take into account punctuation
#sentences_lengths = [len([word for word in word_tokenize(sentence) if word not in string.punctuation]) for sentence in sentences]

average_length = np.mean(sentences_lengths)
print("Average length of a sentence: {}".format(average_length))

#ex4 - Find the collocations in the text (bigram and trigram).
# without applying lower and removing the punctuation. Use 
words_with_punctuation = word_tokenize(content)
bigram_collocation = BigramCollocationFinder.from_words(words_with_punctuation)
trigram_collocation = TrigramCollocationFinder.from_words(words_with_punctuation)
print("\n[with punctuation] Bigrams: ", bigram_collocation.nbest(BigramAssocMeasures.likelihood_ratio, n=100000000))
print("[with punctuation] Trigrams: ", trigram_collocation.nbest(TrigramAssocMeasures.likelihood_ratio, n=100000000))

# when applying first lower() on each word and remove punctuation. Use n=100mil to cope with all the possible bigram/trigram.
words_without_punctuation = [word for word in word_tokenize(content.lower()) if word not in string.punctuation]
# remove stop words
#stop_words = stopwords.words('english')
#words_without_punctuation = [word for word in words_without_punctuation if word not in stop_words]

bigram_collocation = BigramCollocationFinder.from_words(words_without_punctuation)
trigram_collocation = TrigramCollocationFinder.from_words(words_without_punctuation)
print("[without punctuation + lower] Bigrams: ", bigram_collocation.nbest(BigramAssocMeasures.likelihood_ratio, n=100000000))
print("[without punctuation + lower] Trigrams: ", trigram_collocation.nbest(TrigramAssocMeasures.likelihood_ratio, n=100000000))

# ex5 - all words (lowercase) without punctuation
words_without_punctuation = [word for word in word_tokenize(content.lower()) if word not in string.punctuation]

# remove duplicates
words_from_text = set(words_without_punctuation)
print("\nUnique words from text(lower) without punctuation: ", words_from_text)

# ex6 - first N most frequent words
dictionary = {}
for word in words_without_punctuation:
	dictionary[word] = dictionary.get(word, 0) + 1
dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}

N, cnt = 100, 0
print("\nMost used {} words: \n".format(N))
for key in dictionary:
	if cnt >= N:
		break
	if key.isalnum():
		cnt += 1
		print(key, dictionary[key])

# ex 7 - remove stop-words
stop_words = stopwords.words('english')
lws = [word for word in words_without_punctuation if word not in stop_words]
print("\nText without stopwords: ", lws)

# ex 8 - porter stemming
porter_stemmer = nltk.PorterStemmer()
print([porter_stemmer.stem(word) for word in lws][:200])

# ex 9 - Comparison of Porter, Lancaster, Snowball
porter_stemmer = nltk.PorterStemmer()
lancaster_stemmer =nltk.LancasterStemmer()
snowball_stemmer = nltk.SnowballStemmer("english")

porter_list, lancaster_list, snowball_list = [], [], []
NW = 500
visited = {}

for word in lws[:NW]:
	if word not in visited:
		visited[word] = True
		aux = [porter_stemmer.stem(word), lancaster_stemmer.stem(word), snowball_stemmer.stem(word)]
		
		# if stemmer gives different results
		if len(set(aux)) > 1:
			porter_list.append(aux[0])
			lancaster_list.append(aux[1])
			snowball_list.append(aux[2])


print("\nPorter|Lancaster|Snowball")
for i in range(len(porter_list)):
	print("{}|{}|{}".format(porter_list[i], lancaster_list[i], snowball_list[i]))

# ex 10 - Comparison of Snowball vs WordNetLemmatizer
snowball_stemmer = nltk.SnowballStemmer("english")
word_lemmatizer = WordNetLemmatizer()

snowball_list, wordnet_list = [], []
NW = 500
visited = {}

for word in lws[:NW]:
	if word not in visited:
		visited[word] = True
		aux = [snowball_stemmer.stem(word), word_lemmatizer.lemmatize(word)] 
		
		# if the stemmer gives different results
		if len(set(aux)) > 1:
			snowball_list.append(aux[0])
			wordnet_list.append(aux[1])

print("\nSnowball|WordNet")
for i in range(len(porter_list)):
	print("{}|{}".format(snowball_list[i], wordnet_list[i]))

# ex 11 - Most frequent N - lemmas
word_lemmatizer = WordNetLemmatizer()
dictionary = {}
for word in lws:
	lemma = word_lemmatizer.lemmatize(word)
	dictionary[lemma] = dictionary.get(lemma, 0) + 1
dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}

n, cnt = 100, 0
print("\nMost frequent {} lemmas: \n".format(n))
for key in dictionary:
	if cnt >= n:
		break
	if key.isalnum():
		cnt += 1
		print(key, dictionary[key])

# ex 12 - Change number from lws into words
no_changes, N = 0, 10
lws_with_converted_numbers, idx = [], None
for word in lws:
	if word.isnumeric():
		no_changes += 1
		for x in num2words(word):
			lws_with_converted_numbers.append(x)
		if no_changes == N:
			idx = len(lws_with_converted_numbers)
	else:
		lws_with_converted_numbers.append(word)
	

print("\nNumber of changes: {}".format(no_changes))
if idx is not None:
	print("Portion of the list for the first {} changes: {}".format(N, idx/len(lws_with_converted_numbers)))
else:
	print("Portion of the list for the first {} changes: {}".format(N, len(lws_with_converted_numbers)/len(lws_with_converted_numbers)))


# ex 13 - Word inflection
def is_inflection(word1, word2):
	porter_stemmer = nltk.PorterStemmer()
	lancaster_stemmer =nltk.LancasterStemmer()
	snowball_stemmer = nltk.SnowballStemmer("english")
	word_lemmatizer = WordNetLemmatizer()

	if porter_stemmer.stem(word1) == porter_stemmer.stem(word2):
		return True
	if lancaster_stemmer.stem(word1) == lancaster_stemmer.stem(word2):
		return True
	if snowball_stemmer.stem(word1) == snowball_stemmer.stem(word2):
		return True
	if word_lemmatizer.lemmatize(word1) == word_lemmatizer.lemmatize(word2):
		return True

	return False


def concordance_data_from_phrase(word, N, phrase):
	if N % 2 == 0:
		raise Exception("N cannot be even")
	
	stop_words = stopwords.words('english')
	words = [word for word in word_tokenize(phrase.lower()) if word not in string.punctuation and word not in stop_words]
	res = []
	left_no, right_no = N // 2, N // 2
	for i in range(left_no, len(words) - right_no):
		if is_inflection(word, words[i]):
			res.append(words[i - left_no: i + right_no + 1])
	return res
			
def concordance_data_from_text(word, N, text):
	if N % 2 == 0:
		raise Exception("N cannot be even")
	
	sentences = sent_tokenize(text)
	res = []
	for sentence in sentences:
		res = res + concordance_data_from_phrase(word, N, sentence)
	return res


# Example
sentence = "I have two dogs and a cat. Do you have pets too? My cat likes to chase mice. My dogs like to chase my cat."
print(concordance_data_from_phrase("cat", 3, sentence))
print(concordance_data_from_text("cat", 3, sentence))