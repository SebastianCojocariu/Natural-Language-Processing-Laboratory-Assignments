# Cojocariu Sebastian - 407 AI

from nltk.corpus import wordnet
from collections import deque

a = wordnet.synsets('school')
print(a)


# ex1 - Print glosses
def print_glosses(word):
	print("\n#### Exercise 1 #####\n")
	for sset in wordnet.synsets(word):
		print("{} => {}".format(sset, sset.definition()))

print_glosses("school")

# ex2 - can be synonyms (print all the possible glosses - not just one)
# if only one is needed, then return just the first entry in res
def can_be_synonyms(word1, word2, debug=True):
	res = {}
	for sset1 in wordnet.synsets(word1):
		for sset2 in wordnet.synsets(word2):
			if sset1 == sset2:
				res[sset1] = sset1.definition()
	if debug:
		print("\n#### Exercise 2 #####\n")
		for sset in res:
			print("({}, {}) {} => {}.".format(word1, word2, sset, res[sset]))

		if len(res) == 0:
			print("({}, {}) cannot be synonyms".format(word1, word2))

	return res

can_be_synonyms("school", "building")
can_be_synonyms("school", "educate")

# ex3 - print holonyms and meronyms
def get_holonyms_and_meronyms(sset, debug=True):
	holonyms = sset.substance_holonyms() + sset.part_holonyms() + sset.member_holonyms()
	meronyms = sset.substance_meronyms() + sset.part_meronyms() + sset.member_meronyms()
	if debug:
		print("\n#### Exercise 3 #####\n")
		print("Holonyms => {}".format(holonyms))
		print("Meronyms => {}".format(meronyms))

		print("substance_holonyms => {}".format(sset.substance_holonyms()))
		print("part_holonyms => {}".format(sset.part_holonyms()))
		print("member_holonyms => {}".format(sset.member_holonyms()))

		print("substance_meronyms => {}".format(sset.substance_meronyms()))
		print("part_meronyms => {}".format(sset.part_meronyms()))
		print("member_meronyms => {}".format(sset.member_meronyms()))
	return [holonyms, meronyms]

sset = wordnet.synsets("earth")[0]
get_holonyms_and_meronyms(sset)

# ex4 - prints the path of hypernyms
def print_path_hypernyms(sset, debug=True):
	res = [sset]
	current_sset = sset
	while(sset):
		hypernyms_list = current_sset.hypernyms()
		if len(hypernyms_list) == 0:
			break
		[current_sset] = hypernyms_list
		res.append(current_sset)
	
	if debug:
		print("\n#### Exercise 4 #####\n")
		print("{} => {}".format(sset, res))
	return res

sset = wordnet.synsets("school")[0]
print_path_hypernyms(sset)

sset = wordnet.synsets("educate")[0]
print_path_hypernyms(sset)


# ex5 - minimize path to hypernyms
def helper_computer_distances_hypernyms(sset):
	distance_dict, dist = {sset: 0}, 1
	queue = deque([sset])
	
	while(len(queue) > 0):
		aux_queue = deque([])
		while(len(queue) > 0):
			sset = queue.popleft()
			for hyper in sset.hypernyms():
				if hyper not in distance_dict:
					distance_dict[hyper] = dist
					aux_queue.append(hyper)

		dist += 1
		queue = aux_queue
	return distance_dict

def minimze_path_to_hypernyms(sset1, sset2, debug=True):
	distance_dict1 = helper_computer_distances_hypernyms(sset1)
	distance_dict2 = helper_computer_distances_hypernyms(sset2)

	minimum, res = float("Infinity"), []
	for sset in distance_dict1:
		if sset in distance_dict2:
			distance = distance_dict1[sset] + distance_dict2[sset]
			if distance < minimum:
				minimum = distance
				res = [sset]
			elif distance == minimum:
				res.append(sset)
	
	if debug:
		print("\n#### Exercise 5 #####\n")
		print("{}, {} => {}".format(sset1, sset2, res))
	
	return res

sset1 = wordnet.synsets("dog")[0]
sset2 = wordnet.synsets("cat")[0]
minimze_path_to_hypernyms(sset1, sset2)

# ex6 - synset similarity
def synset_similarity(sset, sset_list, debug=True):
	l = []
	for synset in sset_list:
		similarity = sset.path_similarity(synset)
		l.append([similarity, synset])
	
	l.sort(reverse=True)
	
	if debug:
		print("\n#### Exercise 6 #####\n")
		print("Reference synset: {}\n".format(sset))
		for x in l:
			print("Similarity: {} for synset {}".format(x[0], x[1]))
	
	return l

cat_synset = wordnet.synsets("cat")[0]
synsets_list = [wordnet.synsets("animal")[0],
 				wordnet.synsets("tree")[0],
 				wordnet.synsets("house")[0],
 				wordnet.synsets("object")[0],
 				wordnet.synsets("mouse")[0]
 				]
synset_similarity(sset=cat_synset, sset_list=synsets_list)


# ex7 - indirect meronyms (all of them)
def helper_indirect_meronyms(sset):
	visited = {sset: True}
	queue = deque([sset])
	
	while(len(queue) > 0):
		aux_queue = deque([])
		while(len(queue) > 0):
			sset = queue.popleft()
			holonyms, _ = get_holonyms_and_meronyms(sset, debug=False)
			for holonym in holonyms:
				if holonym not in visited:
					visited[holonym] = True
					aux_queue.append(holonym)

		queue = aux_queue
	return visited

def indirect_meronyms(sset1, sset2, debug=True):
	print("\n#### Exercise 7 #####\n")
	holonyms1, holonyms2 = helper_indirect_meronyms(sset1), helper_indirect_meronyms(sset2)

	result = False
	for holonym in holonyms1:
		if holonym in holonyms2:
			result = True
			break
	if debug:
		if result:
			print("{}, {} are indirect meronyms for {}".format(sset1, sset2, holonym))
		else:
			print("{}, {} are not indirect meronyms".format(sset1, sset2))
	
	return False

synset1 = wordnet.synsets("leg")[0]
synset2 = wordnet.synsets("fur")[0]

indirect_meronyms(synset1, synset2)


# ex8 - print antonyms and synonyms
def synonyms_and_antonyms(word):
	print("\n#### Exercise 8 #####\n")
	for sset in wordnet.synsets(word):
		print("Definition: {}".format(sset.definition()))
		print("Synset: {}".format(sset))
		
		antonyms = []
		for sset_antonym in sset.lemmas()[0].antonyms():
			antonyms.append(sset_antonym.name())
		print("Antonyms: {}".format(antonyms))
		
		synonyms = []
		for sset_synonym in sset.similar_tos():
			synonyms.append(sset_synonym.lemma_names())
		print("Synonyms: {}".format(synonyms))
		print("\n")

synonyms_and_antonyms(word="beautiful")


