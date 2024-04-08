import sys
from grammar.pcfg import PCFG


grammar_file = sys.argv[1]
train_file = sys.argv[2]

pcfg = PCFG(grammar_file, "./test.pcfg")

unary_rules = pcfg.grammar.unary_rules
binary_rules = pcfg.grammar.binary_rules

# q = pcfg.estimate(train_file, 5)

print(pcfg.gen_sentence("S"))

# for A, B, C in binary_rules:
#     print(A + " -> " + B + " " + C + " " + str(q.get((A, B, C))))

# for A, w in unary_rules:
#     print(A + " -> " + w + " " + str(q.get((A, w))))

# print(pcfg.grammar.parsable("s s"))
print(pcfg.sentence_prob("s s s s s s d d s s s s d d s"))
