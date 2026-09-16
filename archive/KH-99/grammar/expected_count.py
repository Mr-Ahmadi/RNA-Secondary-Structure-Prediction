from grammar.cfg import CFG
from collections import defaultdict
from grammar.inside_outside import Inside_Outside


class Expected_Count:
    def __init__(self, sentence, grammar: CFG, q: defaultdict):
        self.sentence = sentence.split(" ")
        self.n = len(self.sentence)
        self.grammar = grammar
        self.q = q

        self.f = self.init_f()
        self.io_instance = Inside_Outside(sentence, self.grammar, self.f)
        self.count = self.get_count()

    def get_unary_prob(self, A, w):
        return self.q.get(tuple([A, w]), 0.0)

    def get_binary_prob(self, A, B, C):
        return self.q.get(tuple([A, B, C]), 0.0)

    def init_f(self):
        f = defaultdict(float)

        for i in range(1, self.n + 1):
            for k in range(1, self.n + 1):
                for j in range(1, self.n + 1):
                    for A, B, C in self.grammar.binary_rules:
                        f[tuple([A, B, C, i, k, j])] = self.get_binary_prob(A, B, C)

        for i in range(1, self.n + 1):
            w = self.sentence[i - 1]
            for A in self.grammar.nonterminals:
                if (A, w) in self.grammar.unary_rules:
                    f[tuple([A, i])] = self.get_unary_prob(A, w)
                else:
                    f[tuple([A, i])] = 0.0
                    # self.q[tuple([A, i])] = 0.0
        return f

    def get_count(self):
        count = defaultdict(float)

        for A, B, C in self.grammar.binary_rules:
            for i in range(1, self.n + 1):
                for k in range(i, self.n + 1):
                    for j in range(k + 1, self.n + 1):
                        count[tuple([A, B, C])] += self.io_instance.get_μ_binary(
                            A, B, C, i, k, j
                        )

            count[tuple([A, B, C])] /= self.io_instance.Z

        for A, w in self.grammar.unary_rules:
            for i in range(1, self.n + 1):
                if w == self.sentence[i - 1]:
                    count[tuple([A, w])] += self.io_instance.get_μ_unary(A, i)

            count[tuple([A, w])] /= self.io_instance.Z

        return count
