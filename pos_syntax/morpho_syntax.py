# Cojocariu Sebastian - Group 407 IA

import wikipedia
from nltk.tag.stanford import StanfordPOSTagger
from nltk.corpus import wordnet
import os 
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string 
import pandas as pd
import matplotlib.pyplot as plt

curr_dir = "" # <---- Insert your path here
model_path = "{}/stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger".format(curr_dir)
jar_tagger_path = "{}/stanford-postagger-full-2020-11-17/stanford-postagger.jar".format(curr_dir)


# Exercise 1 
print("####### Exercise 1 #######\n")
document = wikipedia.page("World War II")
content = document.content
# remove trailing spaces, tabs, etc
content = " ".join(content.split())
words = word_tokenize(content)
# words = [word.lower() for word in words]
# words = [word for word in words if word not in stopwords]
# words = [word for word in words if word not in string.punctuation]
words = words[:1000] # choose only 1000 words since more will cause OSError
title = document.title

print("Title: ", document.title)
print("First 200 words: \n", words[:200])

tagger=StanfordPOSTagger(model_path, jar_tagger_path)

pos_tagging = tagger.tag(words[:20])
print("Tagging for the first 20 words: \n", pos_tagging)
print()

# Exercise 2
print("####### Exercise 2 #######\n")
def get_words_with_same_pos_tag(pos_tag, words):
	return [word for (word, part_of_speech) in tagger.tag(words) if part_of_speech == pos_tag]
def get_words_with_same_list_pos_tag(pos_tags, words):
	res = []
	for pos_tag in pos_tags:
		res = res + get_words_with_same_pos_tag(pos_tag, words)
	return res

print(get_words_with_same_pos_tag("NNP", words[:20]))
print()

# Exercise 3
print("####### Exercise 3 #######\n")
pos_tags_nouns = ["NN", "NNS", "NNP", "NNPS"]
nouns = get_words_with_same_list_pos_tag(pos_tags_nouns, words)
print("Nouns: \n", nouns)
print()
pos_tags_verbs = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
verbs = get_words_with_same_list_pos_tag(pos_tags_verbs, words)
print("Verbs: \n", verbs)
print()
print("Percentage of content words (nouns + verbs) in entire text: \n", (len(nouns) + len(verbs)) / len(words))
print()

# Exercise 4
print("####### Exercise 4 #######\n")

def getWordnetPosTag(tag):
	if len(tag) == 0:
		return tag
   
	if tag[0] == "J":
		return wordnet.ADJ
	elif tag[0] == 'V':
		return wordnet.VERB
	elif tag[0] == 'N':
		return wordnet.NOUN
	elif tag[0] == 'R':
		return wordnet.ADV
	else:	
		return wordnet.NOUN

word_lemmatizer = WordNetLemmatizer()
# sentences = sent_tokenize(content.lower())
sentences = sent_tokenize(content)
N = 10
visited = {}
print("Original word | POS | Simple lemmatization | Lemmatization with POS\n")
for i in range(len(sentences[:N])):
	sentence_words = word_tokenize(sentences[i])
	words_tags = tagger.tag(sentence_words) # [(word, tag), ...]
	simple_lemmas = [word_lemmatizer.lemmatize(word) for word in sentence_words]
	lemmas_with_pos = [word_lemmatizer.lemmatize(word, pos=getWordnetPosTag(pos)) for (word, pos) in words_tags]
	
	assert len(simple_lemmas) == len(lemmas_with_pos) and len(words_tags) == len(simple_lemmas) and len(sentence_words) == len(words_tags)
	
	for j in range(len(sentence_words)):
		if sentence_words[j] not in visited:
			visited[sentence_words[j]] = True
			if simple_lemmas[j] != lemmas_with_pos[j]:
				print("{}|{}|{}|{}".format(words_tags[j][0], words_tags[j][1], simple_lemmas[j], lemmas_with_pos[j]))
print()

# Exercise 5
print("####### Exercise 5 #######\n")
part_of_speech_dictionary = {"noun": ["NN", "NNS", "NNP", "NNPS"],
							 "verb": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
							 "adjective": ["JJ", "JJR", "JJS"],
							 "pronoun": ["PRP", "PRP$, WP, WP$"],
							 "adverb": ["RB", "RBR", "RBS", "WRB"],
							 "interjection": ["UH"],
							 "conjuction": ["CC"]}
frequency = []
for part_of_speech in part_of_speech_dictionary:
	aux = get_words_with_same_list_pos_tag(part_of_speech_dictionary[part_of_speech], words)
	frequency.append((len(aux), part_of_speech))

# sort based on frequency

frequency.sort(reverse=True)
K = 5
for i, (freq, part_of_speech) in enumerate(frequency):
	if i < K:
		print("frequency: {}, part of speech: {}".format(freq, part_of_speech))
print()

objects = [pos for (_, pos) in frequency]
frequency = [freq for (freq, _) in frequency]
y_pos = np.arange(len(objects))

plt.bar(y_pos, frequency, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Frequency')
plt.title('Most frequent part of speeches')

plt.show()


# Exercise 6 - recursive descent parsing
print("\n#### Exercise 6 ####\n")
# it is not the examples CFG (it is a modified version)
grammar = nltk.CFG.fromstring("""
	S -> NP VP
	X -> N | A N
	NP -> Pronoun | X | Det X
	VP -> V | V TO VP | V NP | V NP PP 
	PP -> Prep NP 

	V -> "eat" | "like" | "go" | "make"	| "do" | "exist"
	Pronoun -> "I" | "you" | "You" | "she" | "She" | "he" | "He" | "we" | "We" | "they" | "They" | "it" | "It"
	TO -> "to" | "To"
	A -> "beautiful" | "clean"
	Det -> "the" | "The" | "a" | "A" | "an" | "An" | TO
	Prep -> "at" | "in" | "on" | "with" 
	N -> "pie" | "ice-cream" | "spoon" | "soup" | "silver" | "apples"
	""")

sentence = "I like to eat the soup with a clean spoon"
print(sentence)
for tree in nltk.RecursiveDescentParser(grammar).parse(sentence.split()):
	print(tree)


# Exercise 7 - RDP vs SRP
print("\n#### Exercise 7 ####\n")
sentence_to_words1 = "I like to eat the soup with a clean spoon".split()
sentence_to_words2 = "I exist".split()

def helper_print_trees(sentence_to_words, grammar):
	print("-------------")
	print(" ".join(sentence_to_words), "\n")
	rdp_tree, srp_tree = None, None
	
	rdp = nltk.RecursiveDescentParser(grammar)
	for tree in rdp.parse(sentence_to_words):
		rdp_tree = tree
	print("Generated Tree (RDP):\n{}".format(rdp_tree))
	#nltk.app.rdparser()
	
	srp = nltk.ShiftReduceParser(grammar)
	for tree in srp.parse(sentence_to_words):
		srp_tree = tree
	print("Generated Tree (SRP):\n{}".format(srp_tree))

	# check programatically they are the same
	if str(rdp_tree) == str(srp_tree):
		print("The generated trees are the same")
	else:
		print("The generated trees are not the same")
	# nltk.app.srparser()

helper_print_trees(sentence_to_words1, grammar)
helper_print_trees(sentence_to_words2, grammar)
