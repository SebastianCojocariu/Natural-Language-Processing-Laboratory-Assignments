# Cojocariu Sebastian - 407 AI

import nltk
from nltk.corpus import wordnet, stopwords
from nltk.wsd import lesk
from collections import deque
from nltk.tokenize import word_tokenize
from nltk.tag.stanford import StanfordPOSTagger
import string

####### Exercise 1 ####### 
print("##### Exercise 1 ########")
def compute_glosses_similarity(gloss, context, apply_stemmer=False):
	words1, words2 = word_tokenize(gloss.lower()), word_tokenize(context.lower())
	words1 = [word for word in words1 if word not in string.punctuation and word not in stopwords.words("english") and len(word) > 0 and word != "''" and word != "``"]
	words2 = [word for word in words2 if word not in string.punctuation and word not in stopwords.words("english") and len(word) > 0 and word != "''" and word != "``"]

	if apply_stemmer:
		ps = nltk.PorterStemmer()
		words1 = [ps.stem(word) for word in words1]
		words2 = [ps.stem(word) for word in words2]
	
	score = 0
	for w1 in words1:
		for w2 in words2:
			if w1 == w2:
				score += 1

	return score

def original_lesk(target_word, text, apply_stemmer=False):
	text = text.lower()
	words_text = [word for word in word_tokenize(text) if word not in string.punctuation and word not in stopwords.words("english") and len(word) > 0 and word != "''" and word != "``"]
	
	if apply_stemmer:
		ps = nltk.PorterStemmer()
		target_word = ps.stem(target_word)
		words_text = [ps.stem(word) for word in words_text]
	
	#print(words_text)
	
	best_sense, maximum = None, float("-Infinity")
	for target_ss in wordnet.synsets(target_word):
		# original lesk uses context words instead of their synsets definitions. Exclude the target word from the context words.
		score = compute_glosses_similarity(target_ss.definition(), " ".join([word for word in words_text if word != target_word]))
		if score > maximum:
			maximum = score
			best_sense = target_ss
	return "{} -> {}".format(best_sense, best_sense.definition())

phrase = 'Students enjoy going to school, studying and reading books'
word = 'school'
print("Results on '{}' for word '{}'".format(phrase, word))
print("NLTK lesk: {}".format(lesk(nltk.word_tokenize(phrase), word, 'n')))
print("Original lesk: {}".format(original_lesk(word, phrase, apply_stemmer=True)))
print()

####### Exercise 2 ####### 
print("##### Exercise 2 ########")
curr_dir = ""
model_path = "{}/stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger".format(curr_dir)
jar_tagger_path = "{}/stanford-postagger-full-2020-11-17/stanford-postagger.jar".format(curr_dir)

tagger = StanfordPOSTagger(model_path, jar_tagger_path)


def get_extended_glosses(sset):
	hypernyms = list(set([ss.definition() for ss in sset.hypernyms()]))
	hyponyms = list(set([ss.definition() for ss in sset.hyponyms()]))
	meronyms = list(set([ss.definition() for ss in sset.substance_meronyms() + sset.part_meronyms() + sset.member_meronyms()]))
	holonyms = list(set([ss.definition() for ss in sset.substance_holonyms() + sset.part_holonyms() + sset.member_holonyms()]))
	troponyms = list(set([ss.definition() for ss in sset.entailments() + sset.verb_groups() + sset.hypernyms() + sset.hyponyms()]))
	attributes = list(set([ss.definition() for ss in sset.attributes()]))
	similar_to = list(set([ss.definition() for ss in sset.similar_tos()]))
	also_see = list(set([ss.definition() for ss in sset.also_sees()]))

	all_definitions = list(set(hypernyms + hyponyms + meronyms + holonyms + troponyms + attributes + similar_to + also_see + [sset.definition()]))

	return {"hypernyms": hypernyms,
			"hyponyms": hyponyms,
			"meronyms": meronyms,
			"holonyms": holonyms,
			"troponyms": troponyms,
			"attributes": attributes,
			"similar_to": similar_to,
			"also_see": also_see,
			"all_definitions": all_definitions
			}

def extended_compute_glosses_similarity(gloss1, gloss2, apply_stemmer=False):
	words1, words2 = word_tokenize(gloss1.lower()), word_tokenize(gloss2.lower())
	words1 = [word for word in words1 if word not in string.punctuation and word not in stopwords.words("english") and len(word) > 0 and len(word) > 0 and word != "''" and word != "``"]
	words2 = [word for word in words2 if word not in string.punctuation and word not in stopwords.words("english") and len(word) > 0 and len(word) > 0 and word != "''" and word != "``"]

	if apply_stemmer:
		ps = nltk.PorterStemmer()
		words1 = [ps.stem(word) for word in words1]
		words2 = [ps.stem(word) for word in words2]

	# Dynammic programming to find the subsequence overlapping of 2 strings
	m, n = len(words1), len(words2)
	dp = [[0] * n for _ in range(m)]

	for i in range(m):
		dp[i][0] = 1 if words1[i] == words2[0] else 0
	for j in range(n):
		dp[0][j] = 1 if words1[0] == words2[j] else 0
	
	for i in range(1, m):
		for j in range(1, n):
			dp[i][j] = 1 + dp[i - 1][j - 1] if words1[i] == words2[j] else 0

	score = 0
	for i in range(m):
		for j in range(n):
			if not (i + 1 < m and j + 1 < n) or dp[i][j] >= dp[i + 1][j + 1]:
				score += (dp[i][j] ** 2)
	return score


def get_wordnet_pos_tag(tag):
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

def extended_lesk(target_word, text, apply_stemmer=False):
	text = text.lower()
	words_text = [word for word in word_tokenize(text) if word not in string.punctuation and word not in stopwords.words("english") and len(word) > 0 and len(word) > 0 and word != "''" and word != "``"]
	
	if apply_stemmer:
		ps = nltk.PorterStemmer()
		target_word = ps.stem(target_word)
		words_text = [ps.stem(word) for word in words_text]
	
	words_tags = tagger.tag(words_text) # [(word, tag), ...]
	words_tags = [(w, get_wordnet_pos_tag(tag)) for (w, tag) in words_tags]
	context_synsets = [(w, [(ss, get_extended_glosses(ss)["all_definitions"]) for ss in wordnet.synsets(w, pos=tag)]) for (w, tag) in words_tags]

	target_part_of_speech_tag = "n"
	for (w, tag) in words_tags:
		if w == target_word:
			target_part_of_speech_tag = tag
			break

	target_synsets_list = [(ss, get_extended_glosses(ss)["all_definitions"]) for ss in wordnet.synsets(target_word, pos=target_part_of_speech_tag)]
	
	best_sense, maximum = None, float("-Infinity")
	for (target_ss, target_definitions) in target_synsets_list:
		score = 0
		for (w, context_ss_list) in context_synsets:
			if w != target_word:
				for (_, w_ss) in context_ss_list:
					for a in w_ss:
						for b in target_definitions:
							score += extended_compute_glosses_similarity(a, b)
		
		if score > maximum:
			maximum = score
			best_sense = target_ss
	return "{} -> {}".format(best_sense, best_sense.definition())


similarity = extended_compute_glosses_similarity('a1 a2 a3 a4 x4', 'x1 a1 a2 x2 a3 a4 x3 x4')
print("Similarity test for 'a1 a2 a3 a4 x4' and 'x1 a1 a2 x2 a3 a4 x3 x4': {}".format(similarity)) # 9, because [a1, a2] => 2^2, [a3, a4] => 2^ 2, [x4] => 1^2 = 1
print()

print("[line 179] Uncomment this to see the results when using extended_compute_glosses_similarity on different synsets.")
print()
# Uncomment this to see the results when using extended_compute_glosses_similarity on different synsets.

''' 
synsets_list = [wordnet.synsets('student')[0].definition(),
				wordnet.synsets('student')[1].definition(),
				wordnet.synsets('school')[0].definition(),
				wordnet.synsets('school')[1].definition(),
				wordnet.synsets('child')[0].definition(),
				wordnet.synsets('house')[0].definition()
				]

for i in range(len(synsets_list)):
	for j in range(i + 1, len(synsets_list)):
		similarity = extended_compute_glosses_similarity(synsets_list[i], synsets_list[j])
		print("######")
		print("Def1: {}".format(synsets_list[i]))
		print("Def2: {}".format(synsets_list[j]))
		print("Similarity: {}".format(similarity))
'''

phrase = 'Students enjoy going to school, studying and reading books'
word = 'school'
print("Results on '{}' for word '{}'".format(phrase, word))
print("NLTK lesk: {}".format(lesk(nltk.word_tokenize(phrase), word, 'n')))
print("Extended lesk: {}".format(extended_lesk(word, phrase, apply_stemmer=True)))
print()

phrase = 'The student died in the accident.'
word = 'student'
print("An example when extended lesk gives a consistent result with nltk implementation, but original lesk dont.")
print("Result on '{}' for word '{}'".format(phrase, word))
print("NLTK lesk: {}".format(lesk(nltk.word_tokenize(phrase), word, 'n')))
print("Original lesk: {}".format(original_lesk(word, phrase, apply_stemmer=True)))
print("Extended lesk: {}".format(extended_lesk(word, phrase, apply_stemmer=True)))
print()

