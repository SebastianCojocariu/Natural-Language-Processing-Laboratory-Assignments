# Cojocariu Sebastian - 407 AI

import gensim 
import sklearn
import nltk
from gensim.models import Word2Vec 
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from sklearn.cluster import KMeans
from nltk.corpus import wordnet, stopwords
import numpy as np
import urllib
import string


GoogleNews_PATH = "" # <---- Insert your path here

model = gensim.models.KeyedVectors.load_word2vec_format(GoogleNews_PATH, binary=True)

# ex1 - print number of words
print("#### Exercise 1 ####\n")
assert len(model.vocab) == model.vectors.shape[0], "model.vocab length is different from the number of embeddings"
print("Length of vocabulary: {}\n".format(len(model.vocab)))


print("#### Exercise 2 ####\n")
text = 'World War II is generally considered to have begun on 1 September 1939, when Nazi Germany, under Adolf Hitler, invaded Poland. The United Kingdom and France subsequently declared war on the 3rd. Under the Molotov–Ribbentrop Pact of August 1939, Germany and the Soviet Union had partitioned Poland and marked out their "spheres of influence" across Finland, Romania and the Baltic states. From late 1939 to early 1941, in a series of campaigns and treaties, Germany conquered or controlled much of continental Europe, and formed the Axis alliance with Italy and Japan (along with other countries later on). Following the onset of campaigns in North Africa and East Africa, and the fall of France in mid-1940, the war continued primarily between the European Axis powers and the British Empire, with war in the Balkans, the aerial Battle of Britain, the Blitz of the UK, and the Battle of the Atlantic. On 22 June 1941, Germany led the European Axis powers in an invasion of the Soviet Union, opening the Eastern Front, the largest land theatre of war in history and trapping the Axis powers, crucially the German Wehrmacht, in a war of attrition.'

# ex2 - find missing words
words_matrix = [[word for word in word_tokenize(sentence) if word.lower() not in string.punctuation and word.lower() not in stopwords.words("english") and len(word) > 0 and word != "''" and word != "``"] for sentence in sent_tokenize(text)]

# dictionary that stores which words dont appear in word2vec + their number of appereances
non_existent_words = defaultdict(lambda: 0)
for splitted_phrase in words_matrix:
	for word in splitted_phrase:
		if word not in model.vocab and len(word) > 0:
			non_existent_words[word] += 1


print("Non-existent words in word2vec, but which exist in text:\n")
for x in non_existent_words:
	print("'{}' with {} appereances".format(x, non_existent_words[x]))
print()
# ex3 - distant/closest words
print("#### Exercise 3 ####\n")

all_words = {}
for splitted_phrase in words_matrix:
	for word in splitted_phrase:
		if word in model.vocab:
			all_words[word] = True

similarity_list = []
for w1 in all_words:
	for w2 in all_words:
		if w1 != w2:
			similarity = model.similarity(w1, w2)
			similarity_list.append((similarity, w1, w2))

similarity_list.sort(key=lambda x: x[0])
print("The two most distant words are '{}' and '{}'. Their similariy is: {}".format(similarity_list[0][1], similarity_list[0][2], similarity_list[0][0]))
print("The two closest words are '{}' and '{}'. Their similariy is: {}".format(similarity_list[-1][1], similarity_list[-1][2], similarity_list[-1][0]))
print()
# ex4 - Names Entity Recognition
print("#### Exercise 4 ####\n")
for sentence in nltk.sent_tokenize(text):
	for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence))):
		if hasattr(chunk, 'label'):
			named_entity_lower = ' '.join([c[0].lower() for c in chunk])
			named_entity_upper = ' '.join([c[0] for c in chunk])
		
			if named_entity_lower in model.vocab:
			 	similarity_list = []
			 	for word in all_words:
			 		if word != named_entity_lower:
				 		similarity = model.similarity(word, named_entity_lower)
				 		similarity_list.append((similarity, word))
			 	similarity_list.sort(key=lambda x: x[0], reverse=True)
			 	print("[Lower] {} => {}".format(named_entity_lower, [x[1] for x in similarity_list[:5]]))

			if named_entity_upper in model.vocab:
			 	similarity_list = []
			 	for word in all_words:
			 		if word != named_entity_upper:
				 		similarity = model.similarity(word, named_entity_upper)
				 		similarity_list.append((similarity, word))
			 	similarity_list.sort(key=lambda x: x[0], reverse=True)
			 	print("[Upper] {} => {}".format(named_entity_upper, [x[1] for x in similarity_list[:5]]))
print()
# ex5 - KMeans
print("#### Exercise 5 ####\n")
X_train = []
for word in all_words:
	X_train.append(model.get_vector(word))
X_train = np.asarray(X_train)
print("The shape of training data: ".format(X_train.shape))

kmeans = KMeans(n_clusters=5, random_state=0).fit(X_train)
labels = kmeans.labels_

for idx_clst, label in enumerate(set(labels)):
	cluster = []
	for i, word in enumerate(all_words):
		if labels[i] == label:
			cluster.append(word)
	print("Cluster {} top 10 representatives: {}".format(idx_clst, cluster[:10]))
print()

