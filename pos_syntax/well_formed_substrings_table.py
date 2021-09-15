# Cojocariu Sebastian - 407 AI

import nltk

# add all rules X such that exists X -> AB
def production_case(A, B, chomsky_grammar):
  res = []
  for prod in chomsky_grammar.productions():
    if prod.is_nonlexical():
      if len(prod.rhs()) == 2:
        if (A, B) == prod.rhs():
          res.append(prod.lhs())
  return res

# add all rules X such that exists X -> terminal
def terminal_case(terminal, chomsky_grammar):
  res = []
  for prod in chomsky_grammar.productions():
    if not prod.is_nonlexical():
      if len(prod.rhs()) == 1:
        if terminal == prod.rhs()[0]:
          res.append(prod.lhs())
  return res

def substring_tables(string, grammar):
  # transform into chomsky normal form 
  # only A -> B C and A -> a types (no epsilon rules A -> epsilon)
  chomsky_grammar = grammar.chomsky_normal_form()
  sentence_to_words = string.split()
  no_words = len(sentence_to_words)

  # dp[(i, j)] = the root of the subtree containing all the words from i to j (inclusively)
  # dp[(i, i)] = {X| such that exists production X -> sentence_to_words[i]} 
  dp = {(i, j): [] for i in range(no_words) for j in range(i, no_words)}

  for dist in range(no_words):
    for i in range(no_words - dist):
      j = i + dist
      if dist == 0:
        dp[(i, j)] = terminal_case(sentence_to_words[i], chomsky_grammar)
      else:
        res = []
        for k in range(i, j):
          for A in dp[(i, k)]:
            for B in dp[(k + 1, j)]:
              res = res + production_case(A, B, chomsky_grammar)
        dp[(i, j)] = list(set(res))

  for i in range(no_words):
    line = [dp[(i, j)] for j in range(i, no_words)]
    # add fake empty [] because there will be no solutions for dp[(i, j)] with j > i
    line = [[]] * i + line
    print(line)

  if chomsky_grammar.start() in dp[(0, no_words - 1)]:
    print("The phrase: {} is part of the grammar.".format(string))
  else:
    print("The phrase: {} is not part of the grammar.".format(string))

grammar1 = nltk.CFG.fromstring(""" 
  S -> NP VP
  S -> VP
  NP -> DT NN
  NP -> DT JJ NN
  NP -> PRP
  VP -> VBP NP
  VP -> VBP VP
  VP -> VBG NP
  VP -> TO VP
  VP -> VB
  VP -> VB NP
  NN -> "show" | "book"
  PRP -> "I"
  VBP -> "am"
  VBG -> "watching"
  VB -> "show"
  DT -> "a" | "the"
  MD -> "will"
""") 

string1 = 'I am watching a show'
substring_tables(string1, grammar1)


grammar2 = nltk.CFG.fromstring("""
  S -> NP VP
  NP -> PRP
  NP -> NP CC NP
  NP -> NNS
  VP -> VBP S
  VP -> VBP ADJP
  ADJP -> JJ PP
  PP -> IN NP
  PRP -> "I" | "it"
  VBP -> "collect" | "'m"
  NNS -> "ants"
  CC -> "and"
  JJ -> "proud"
  IN -> "of" 
  """)
string2 = 'I collect ants and I\'m proud of it'
substring_tables(string2, grammar2)

