# Cojocariu Sebastian - 407 AI

import os
from nltk.parse.stanford import StanfordDependencyParser, StanfordParser

os.environ['JAVAHOME'] = "/usr/bin/java"
os.environ['STANFORD_PARSER'] = parser_path = 'stanford-parser-full-2020-11-17/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = models_path = 'stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar'

dependency_parser = StanfordDependencyParser(path_to_jar='stanford-parser-full-2020-11-17/stanford-parser.jar', path_to_models_jar='stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar')
parser = StanfordParser(model_path="stanford-parser-full-2020-11-17/englishPCFG.ser.gz")

sentences = open("ex5_input.txt", "r").read().splitlines()
sentences_parsed = list(parser.raw_parse_sents(sentences))
dependencies_parsed = list(dependency_parser.raw_parse_sents(sentences))

with open("ex5_output.txt", "w+") as f:
	for i in range(len(sentences)):
		f.write("Sentence - number {}\n".format(i))
		f.write("{}\n".format(sentences[i]))
		f.write("{}\n".format(str(list(sentences_parsed[i])), '\n'))
		for dep in dependencies_parsed[i]:
			f.write("{}\n".format(str(list(dep.triples())), '\n'))
		f.write("------------------------\n")