import random
from grammar.cfg import CFG
from collections import defaultdict
from grammar.expected_count import Expected_Count


class PCFG:
    def __init__(self, grammar_file: str, train_file: str):
        self.sentences = self.read_train_file(train_file)
        self.grammar = CFG(grammar_file)
        self.q = self.init_q()

    def read_train_file(self, filename: str):
        sentences = []
        with open(filename) as f:
            for line in f.readlines():
                if len(line):
                    sentences.append(line.strip())
        return sentences

    def init_q(self):
        q = defaultdict(float)

        for A in self.grammar.nonterminals:
            c = 0
            for A2, w in self.grammar.unary_rules:
                if A2 == A:
                    c = c + 1

            for A2, B, C in self.grammar.binary_rules:
                if A2 == A:
                    c = c + 1

            sum = 0.0
            for A2, w in self.grammar.unary_rules:
                if A2 == A:
                    if c == 1:
                        q[tuple([A, w])] = 1.0 - sum
                    else:
                        q[tuple([A, w])] = random.uniform(0, 1.0 - sum)

                    c = c - 1
                    sum = sum + q[tuple([A, w])]

            for A2, B, C in self.grammar.binary_rules:
                if A2 == A:
                    if c == 1:
                        q[tuple([A, B, C])] = 1.0 - sum
                    else:
                        q[tuple([A, B, C])] = random.uniform(0, 1.0 - sum)

                    c = c - 1
                    sum = sum + q[tuple([A, B, C])]
        return q

    def estimate(self, iter_num=20):
        q = self.q

        for itration in range(1, iter_num + 1):
            print("Itration number:", itration)

            f = defaultdict(float)

            for A, w in self.grammar.unary_rules:
                f[tuple([A, w])] = 0.0

            for A, B, C in self.grammar.binary_rules:
                f[tuple([A, B, C])] = 0.0

            for i in range(0, len(self.sentences)):
                sentence = self.sentences[i]
                ec_instance = Expected_Count(sentence, self.grammar, q)
                count = ec_instance.get_count()

                for A, w in self.grammar.unary_rules:
                    f[tuple([A, w])] += count.get(tuple([A, w]))
                for A, B, C in self.grammar.binary_rules:
                    f[tuple([A, B, C])] += count.get(tuple([A, B, C]))

            for A in self.grammar.nonterminals:
                sum_f = 0.0
                for A2, w in self.grammar.unary_rules:
                    if A2 == A:
                        sum_f += f[tuple([A, w])]

                for A2, B, C in self.grammar.binary_rules:
                    if A2 == A:
                        sum_f += f[tuple([A, B, C])]

                for A2, w in self.grammar.unary_rules:
                    if A2 == A and sum_f:
                        q[tuple([A, w])] = f[tuple([A, w])] / sum_f
                    if A2 == A and f[tuple([A, w])] == 0.0:
                        q[tuple([A, w])] = 0.0

                for A2, B, C in self.grammar.binary_rules:
                    if A2 == A and sum_f:
                        q[tuple([A, B, C])] = f[tuple([A, B, C])] / sum_f
                    if A2 == A and f[tuple([A, B, C])] == 0.0:
                        q[tuple([A, B, C])] = 0.0

        print("Estimation complete!")

        return q

    def sentence_prob(self, sentence: str):
        P = defaultdict(float)

        sentence = sentence.split(" ")
        length = len(sentence)

        for i in range(1, length + 1):
            for A, w in self.grammar.unary_rules:
                if w == sentence[i - 1]:
                    P[i, i, A] = self.q.get((A, w))

        for l in range(2, length + 1):
            for i in range(1, length + 2 - l):
                j = i + l - 1
                for k in range(i, j):
                    for A, B, C in self.grammar.binary_rules:
                        if P[i, k, B] and P[k + 1, j, C]:
                            P[i, j, A] = max(
                                (P[i, k, B] * self.q.get((A, B, C)) * P[k + 1, j, C]),
                                P[i, j, A],
                            )

        return P[1, length, "S"]

    def gen_sentence(self, symbol):
        tokens = []
        for A, w in self.grammar.unary_rules:
            if A == symbol:
                num = int(self.q.get((A, w)) * 1000)
                for _ in range(num):
                    tokens.append(w)
        for A, B, C in self.grammar.binary_rules:
            if A == symbol:
                num = int(self.q.get((A, B, C)) * 1000)
                for _ in range(num):
                    tokens.append(tuple([B, C]))
        inx = int(random.uniform(0, len(tokens)))
        next_symbol = tokens[inx]
        if isinstance(next_symbol, tuple):
            return (
                self.gen_sentence(next_symbol[0])
                + " "
                + self.gen_sentence(next_symbol[1])
            )
        else:
            return next_symbol
