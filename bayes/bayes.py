# Cojocariu Sebastian - 407 AI

import nltk
from nltk.classify import accuracy
import random
from nltk.corpus import senseval, wordnet, stopwords
import string
from prettytable import PrettyTable

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

def get_features(senseval_instance, window_size):
	position = senseval_instance.position
	context = senseval_instance.context
	sense = senseval_instance.senses[0]
	
	# it should be only one sense
	assert len(senseval_instance.senses) == 1, "{}".format(senseval_instance.senses)
	
	features = {}
	features["initial_position_idx"] = position
	features["len_context"] = len(context)

	# remove stop words or punctuation
	new_context, new_position = [], position
	for i in range(len(context)):
		if not isinstance(context[i], (tuple, list)):
			if i < position:
				new_position -= 1
		elif context[i][0].lower() in string.punctuation or context[i][0].lower() in stopwords.words("english") or context[i][0] == "``" or context[i][0] == "''" or len(context[i][0]) == 0:
			if i < position:
				new_position -= 1
		else:
			new_context.append(context[i])

	new_context = [(0, "None1")] * (window_size // 2) + new_context + [(0, "None2")] * (window_size // 2)
	new_position += window_size // 2

	features["new_position_idx"] = new_position
	features["len_context_after_removal"] = len(new_context)

	for i in range(new_position - window_size // 2, new_position + window_size // 2 + 1):
		tag = new_context[i][1]
		reduced_tag = get_wordnet_pos_tag(tag)

		# the part of speech surrounding the target word (in the window size)
		features["part_of_speech_{}".format(i - new_position)] = tag
		features["reduced_part_of_speech_{}".format(i - new_position)] = reduced_tag

		# the number of different tags (in the window size)
		features["{}".format(tag)] = features.get("{}".format(tag), 0) + 1 
		features["{}".format(reduced_tag)] = features.get("{}".format(reduced_tag), 0) + 1 

	# These features didnt worked as well as expected
	'''
	for i in range(window_size//2, len(new_context) - window_size // 2):
		tag = new_context[i][1]
		reduced_tag = get_wordnet_pos_tag(tag)

		features["{}".format(tag)] = features.get("{}".format(tag), 0) + 1 
		features["{}".format(reduced_tag)] = features.get("{}".format(reduced_tag), 0) + 1 
	'''

	return (features, sense)

for word in ["serve", "line", "interest"]:
	inst = senseval.instances("{}.pos".format(word))

	# compute features from context + shuffle
	SIZE_SPLIT = 0.9
	
	all_data = [get_features(inst[i], window_size=11) for i in range(len(inst))]
	random.shuffle(all_data)
	train, validation = all_data[:int(SIZE_SPLIT * len(all_data))], all_data[int(SIZE_SPLIT * len(all_data)):]

	model = nltk.NaiveBayesClassifier.train(train)

	with open("./naiveBayes_for_{}".format(word), "w+") as f:
		f.write("###################################\n")
		f.write("Word: {}\n".format(word))
		f.write("Data length: {}\n".format(len(all_data)))
		f.write("Senses available in data: {}\n".format(set([sense for (_, sense) in all_data])))
		f.write("Accuracy on train: {}\n".format(accuracy(model, train)))
		f.write("Accuracy on validation: {}\n".format(accuracy(model, validation)))

		table = PrettyTable()
		table.field_names = ["predicted", "truth"]
		for i in range(min(100, len(validation))):
			predicted_label = model.classify(validation[i][0])
			true_label = validation[i][1]
			table.add_row([predicted_label, true_label])
		f.write(str(table))
