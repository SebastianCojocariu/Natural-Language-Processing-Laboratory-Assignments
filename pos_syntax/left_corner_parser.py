# Cojocariu Sebastian - 407 AI

import os 
import nltk
from copy import deepcopy
from nltk import Nonterminal, nonterminals, Production, CFG

# some helper functions that I did not used in the final algorithm as i found other strategy
'''
# generate left corner table as per laboratory requirements
def generate_left_corner_table(grammar):
	productions = grammar.productions()
	table = {}
	for prod in productions:
		lhs, rhs = prod.lhs(), prod.rhs()
		first_symbol = rhs if not isinstance(rhs, tuple) else rhs[0]
		if lhs not in table:
			table[lhs] = [first_symbol]
		else:
			if first_symbol not in table[lhs]:
				table[lhs].append(first_symbol)
	return table

# generate all possible routes from a current_node to a target_node based on left_corner table
def generate_route(target_node, current_node, grammar, lc_table, current_list, res):
	if current_node == target_node:
		res.append(deepcopy(current_list))
	else:
		for left_nonterminal in lc_table:
			if current_node in lc_table[left_nonterminal]:
				current_list.append(left_nonterminal)
				generate_route(target_node, left_nonterminal, grammar, lc_table, current_list, res)
				current_list.pop()

# return all productions from grammar, removing the left_side and the leftmost symbol
def get_production_with_left_symbol(grammar, left_side, leftmost_symbol):
	res = []
	for production in grammar.productions():
		if left_side == production.lhs() and leftmost_symbol == production.rhs()[0]:
			res.append(list(production.rhs())[1:])
	return res
'''

def find_production_with_given_left_corner(grammar, left_corner):
	res = []
	for production in grammar.productions():
		if left_corner == production.rhs()[0]:
			res.append((production.lhs(), list(production.rhs())[1:]))
	return res

# probably it would be more efficient to do it as a BFS starting initializing the queue with the leaves (terminals)
# but it isnt important now.
def populate_closure_table(table):
	def run_dfs_helper(node, table, visited):
		visited[node] = True
		if node in table:
			for neigh in table[node]:
				if neigh not in visited:
					run_dfs_helper(neigh, table, visited)

	# add all the reached nodes in the LC closure of node
	table_lc_closure = {}
	for node in table:
		visited = {}
		run_dfs_helper(node, table, visited)
		table_lc_closure[node] = [reached_node for reached_node in visited]
	return table_lc_closure


def get_closure_leftCornerTable(grammar):
	productions = grammar.productions()
	
	# store left corner parser
	table_lc, table_inverse = {}, {}
	for prod in productions:
		lhs, rhs = prod.lhs(), prod.rhs()
		leftmost_symbol = rhs[0]
		
		if lhs not in table_lc:
			table_lc[lhs] = [leftmost_symbol]
		else:
			if leftmost_symbol not in table_lc[lhs]:
				table_lc[lhs].append(leftmost_symbol)
		
	# store the closure (links that can be reached through other links)
	table_lc_closure = populate_closure_table(table_lc)
	return table_lc_closure


# here the magic happens. References:
# 1 -> Description at https://www.english-linguistics.de/fr/teaching/ws06-07/cl2/slides/LeftCorner-laura.pdf (slide 63 + 134 using LC closure)
# 2 -> https://user.phil-fak.uni-duesseldorf.de/~kallmeyer/Parsing/left-corner.pdf (+ look-ahead using LC closure)
def left_corner_parser_helper(grammar, stack_completed, stack_td, stack_lhs, closure_lc):
	# if already decided that the phrase if part of CFG, do nothing
	# completed, td, lhs
	if len(stack_completed) == 0 and len(stack_td) == 0 and len(stack_lhs) == 0:
		return True

	res = False
	# reduce
	if len(stack_completed) > 0 and len(stack_td) > 0:
		for x in find_production_with_given_left_corner(grammar, stack_completed[0]):
			A, X_list = x[0], x[1]
			# look-ahead condition (LC closure)
			if stack_td[0] in closure_lc and (A in closure_lc[stack_td[0]]):
				res = res or left_corner_parser_helper(grammar, stack_completed[1:], X_list + ["$"] + stack_td, [A] + stack_lhs, closure_lc)
				# to speed up the process (if there is a path, dont recur for the remaining cases)
				if res:
					return res
	# move
	if len(stack_td) > 0 and stack_td[0] == "$" and len(stack_lhs) > 0:
		res = res or left_corner_parser_helper(grammar, [stack_lhs[0]] + stack_completed, stack_td[1:], stack_lhs[1:], closure_lc)
	# to speed up the process (if there is a path, dont recur for the remaining cases)
	if res:
		return res
	# remove
	if len(stack_completed) > 0 and len(stack_td) > 0 and stack_completed[0] == stack_td[0]:
		res = res or left_corner_parser_helper(grammar, stack_completed[1:], stack_td[1:], stack_lhs, closure_lc)
	return res
		
def left_corner_parser(phrase, grammar):
	words = phrase.split()
	stack_completed, stack_td, stack_lhs = words, [grammar.start()], []
	closure_lc = get_closure_leftCornerTable(grammar)
	res = left_corner_parser_helper(grammar, stack_completed, stack_td, stack_lhs, closure_lc)
	if res:
		print("Accepted")
	else:
		print("Rejected")

grammar_simple1 = nltk.CFG.fromstring("""
	S -> 'a' S 'a' | 'b' S 'b' | 'c'
	""")

grammar_simple2 = nltk.CFG.fromstring("""
	S -> A S A | B S B | C
	A -> 'a'
	B -> 'b'
	C -> 'c'
	""")

grammar = nltk.CFG.fromstring("""
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


left_corner_parser("a b c b a", grammar_simple1)
left_corner_parser("a b c b a", grammar_simple2)

print("LC closure => {}".format(get_closure_leftCornerTable(grammar)))
left_corner_parser("I am watching a show", grammar)
