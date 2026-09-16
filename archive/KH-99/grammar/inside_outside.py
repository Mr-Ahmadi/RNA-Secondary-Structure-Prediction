from grammar.cfg import CFG
from collections import defaultdict


class Inside_Outside:
    def __init__(self, sentence, grammar: CFG, f: defaultdict):
        self.sentence = sentence.split(" ")
        self.n = len(self.sentence)
        self.grammar = grammar
        self.f = f
        self.inside = self.get_inside_terms()
        self.outside = self.get_outside_terms()
        self.Z = self.inside["S", 1, self.n]

    def get_unary_rule_prob(self, X, i):
        return self.f.get(tuple([X, i]), 0.0)

    def get_binary_rule_prob(self, X, Y, Z, i, k, j):
        return self.f.get(tuple([X, Y, Z, i, k, j]), 0.0)

    def get_inside_terms(self):
        inside = defaultdict(float)

        for i in range(1, 1 + self.n):
            w = self.sentence[i - 1]
            for X in self.grammar.nonterminals:
                if tuple([X, w]) in self.grammar.unary_rules:
                    inside[X, i, i] = self.get_unary_rule_prob(X, i)
                else:
                    inside[X, i, i] = 0.0

        for l in range(1, self.n):
            for i in range(1, self.n - l + 1):
                j = i + l
                for A, B, C in self.grammar.binary_rules:
                    for k in range(i, j):
                        if inside[B, i, k] and inside[C, k + 1, j]:
                            inside[A, i, j] += (
                                self.get_binary_rule_prob(A, B, C, i, k, j)
                                * inside[B, i, k]
                                * inside[C, k + 1, j]
                            )

        if inside["S", 1, self.n]:
            return inside
        else:
            print(self.sentence)

    def get_outside_terms(self):
        outside = defaultdict(float)

        n = self.n
        outside["S", 1, n] = 1
        for i in range(1, n + 1):
            for j in range(n, 0, -1):
                if i == 1 and j == n:
                    continue
                for B, C, A in self.grammar.binary_rules:
                    for k in range(1, i):
                        outside[A, i, j] += (
                            self.get_binary_rule_prob(B, C, A, k, i - 1, j)
                            * self.inside[C, k, i - 1]
                            * outside[B, k, j]
                        )
                for B, A, C in self.grammar.binary_rules:
                    for k in range(j + 1, n + 1):
                        outside[A, i, j] += (
                            self.get_binary_rule_prob(B, A, C, i, j, k)
                            * self.inside[C, j + 1, k]
                            * outside[B, i, k]
                        )

        return outside

    def get_μ_unary(self, A, i, j=-1):
        j = i if j == -1 else j
        return self.inside[A, i, j] * self.outside[A, i, j]

    def get_μ_binary(self, A, B, C, i, k, j):
        return (
            self.outside[A, i, j]
            * self.get_binary_rule_prob(A, B, C, i, k, j)
            * self.inside[B, i, k]
            * self.inside[C, k + 1, j]
        )
