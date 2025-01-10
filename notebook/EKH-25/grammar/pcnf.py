import random
from grammar.cnf import CNF
from collections import defaultdict
from grammar.expected_count import Expected_Count

def log(x): 
    from math import log
    if x > 0: return log(x)
    else: return float("-inf")

class PCNF:
    def __init__(self, grammar_file: str, probablity_file=""):
        self.grammar = CNF(grammar_file)
        if probablity_file == "":
            self.q = self.init_q()
        else:
            self.q = self.read_pcfg_file(probablity_file)

    def read_pcfg_file(self, filename: str):
        pcfg = []
        with open(filename) as file:
            for line in file:
                line = line.strip().split(" -> ")
                pcfg.append((line[0], line[1]))
        q = defaultdict(float)
        for rule in pcfg:
            tmp = rule[1].split()
            if len(tmp) == 2:
                q[(rule[0], tmp[0])] = float(tmp[1])
            elif len(tmp) == 3:
                q[(rule[0], tmp[0], tmp[1])] = float(tmp[2])
        return q

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
                        q[(A, w)] = 1.0 - sum
                    else:
                        q[(A, w)] = random.uniform(0, 1.0 - sum)

                    c = c - 1
                    sum = sum + q[(A, w)]

            for A2, B, C in self.grammar.binary_rules:
                if A2 == A:
                    if c == 1:
                        q[(A, B, C)] = 1.0 - sum
                    else:
                        q[(A, B, C)] = random.uniform(0, 1.0 - sum)

                    c = c - 1
                    sum = sum + q[(A, B, C)]
        return q

    def estimate(self, train_file: str, iter_num=20):
        q = self.q
        self.sentences = self.read_train_file(train_file)

        for itration in range(1, iter_num + 1):
            print("Itration number:", itration)

            f = defaultdict(float)

            for A, w in self.grammar.unary_rules:
                f[(A, w)] = 0.0

            for A, B, C in self.grammar.binary_rules:
                f[(A, B, C)] = 0.0

            for i in range(0, len(self.sentences)):
                sentence = self.sentences[i]
                ec_instance = Expected_Count(sentence, self.grammar, q)
                count = ec_instance.get_count()

                for A, w in self.grammar.unary_rules:
                    f[(A, w)] += count.get((A, w))
                for A, B, C in self.grammar.binary_rules:
                    f[(A, B, C)] += count.get((A, B, C))

            for A in self.grammar.nonterminals:
                sum_f = 0.0
                for A2, w in self.grammar.unary_rules:
                    if A2 == A:
                        sum_f += f[(A, w)]

                for A2, B, C in self.grammar.binary_rules:
                    if A2 == A:
                        sum_f += f[(A, B, C)]

                for A2, w in self.grammar.unary_rules:
                    if A2 == A and sum_f:
                        q[(A, w)] = f[(A, w)] / sum_f
                    if A2 == A and f[(A, w)] == 0.0:
                        q[(A, w)] = 0.0

                for A2, B, C in self.grammar.binary_rules:
                    if A2 == A and sum_f:
                        q[(A, B, C)] = f[(A, B, C)] / sum_f
                    if A2 == A and f[(A, B, C)] == 0.0:
                        q[(A, B, C)] = 0.0

        print("Estimation complete!")

        return q

    def sentence_prob(self, sentence: str):
        P = defaultdict(float)
        table = defaultdict(None)

        sentence = sentence.strip().split(" ")
        length = len(sentence)

        for i in range(1, length + 1):
            for A, w in self.grammar.unary_rules:
                if w == sentence[i - 1]:
                    P[(i, i, A)] = self.q.get((A, w), 0)
                    table[(i, i, A)] = [(i, i, w)]
                    table[(i, i, w)] = []


        for l in range(2, length + 1):
            for i in range(1, length + 2 - l):
                j = i + l - 1
                for k in range(i, j):
                    for A, B, C in self.grammar.binary_rules:
                        if P.get((i, k, B), 0) and P.get((k + 1, j, C), 0):
                            if (
                                P.get((i, k, B), 0)
                                * self.q.get((A, B, C), 0)
                                * P.get((k + 1, j, C), 0)
                            ) > P.get((i, j, A), 0):
                                P[(i, j, A)] = (
                                    P.get((i, k, B), 0)
                                    * (self.q.get((A, B, C), 0))
                                    * P.get((k + 1, j, C), 0)
                                )
                                table[(i, j, A)] = [(i, k, B), (k + 1, j, C)]

        return P[1, length, "S"], table
    

    def sentence_prob__(self, sentence: str, flag_ratio):
        # Split sentence into words and identify ignored ones
        words = sentence.strip().split(" ")
        filtered_sentence = [word for word in words if word != "$"]
        
        flagged_counts = []
        counter = 0
        for i in range(len(words)):
            if words[i] == "$":
                counter += 1
            else:
                flagged_counts.append(counter)
        
        length = len(filtered_sentence)

        if length == 0:
            return float(0), {}

        P = defaultdict(float)
        table = defaultdict(None)
                    
        for i in range(1, length + 1):
            for A, w in self.grammar.unary_rules:
                if w == filtered_sentence[i - 1]:
                    P[(i, i, A)] = self.q.get((A, w), 0)
                    table[(i, i, A)] = [(i, i, w)]
                    table[(i, i, w)] = []
                    

        # Binary rules (off-diagonal cells)
        for l in range(2, length + 1):  # Length of the span
            for i in range(1, length + 2 - l):  # Start index
                j = i + l - 1  # End index                
                for k in range(i, j):
                    for A, B, C in self.grammar.binary_rules:
                        if P.get((i, k, B), 0) and P.get((k + 1, j, C), 0):
                            if A.startswith("$"):
                                flagged_count = flagged_counts[j - 1] - flagged_counts[i - 1]
                                                                
                                Prob = (P.get((i, k, B), 0)
                                        * self.q.get((A, B, C), 0)
                                        * P.get((k + 1, j, C), 0) * pow(flag_ratio, flagged_count))
                                
                                if Prob > P.get((i, j, A), 0):
                                    P[(i, j, A)] = Prob
                                    table[(i, j, A)] = [(i, k, B), (k + 1, j, C)]
                            else:
                                Prob = (P.get((i, k, B), 0)
                                        * self.q.get((A, B, C), 0)
                                        * P.get((k + 1, j, C), 0)) 
                                
                                if Prob > P.get((i, j, A), 0):
                                    P[(i, j, A)] = Prob
                                    table[(i, j, A)] = [(i, k, B), (k + 1, j, C)]
                            
        return P[1, length, "S"], table
    
    
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
        
        
    # Via log-prob
    # def sentence_prob(self, sentence: str):
    #     P = defaultdict(lambda: float("-inf"))
    #     table = defaultdict(None)

    #     sentence = sentence.strip().split(" ")
    #     length = len(sentence)

    #     for i in range(1, length + 1):
    #         for A, w in self.grammar.unary_rules:
    #             if w == sentence[i - 1]:
    #                 P[(i, i, A)] = log(self.q.get((A, w), 0))
    #                 table[(i, i, A)] = [(i, i, w)]
    #                 table[(i, i, w)] = []


    #     for l in range(2, length + 1):
    #         for i in range(1, length + 2 - l):
    #             j = i + l - 1
    #             for k in range(i, j):
    #                 for A, B, C in self.grammar.binary_rules:
    #                     if (P.get((i, k, B), float("-inf")) != float("-inf") 
    #                     and P.get((k + 1, j, C), float("-inf")) != float("-inf")):
    #                         if (
    #                             P.get((i, k, B), float("-inf"))
    #                             + log(self.q.get((A, B, C), 0))
    #                             + P.get((k + 1, j, C), float("-inf"))
    #                         ) > P.get((i, j, A), float("-inf")):
    #                             P[(i, j, A)] = (
    #                                 P.get((i, k, B), float("-inf"))
    #                                 + log(self.q.get((A, B, C), 0))
    #                                 + P.get((k + 1, j, C), float("-inf"))
    #                             )
    #                             table[(i, j, A)] = [(i, k, B), (k + 1, j, C)]
                                
    #                             if i == 1 and A == "S":
    #                                 print(f"Substring (1, {j}) parsed.")

    #     return P[1, length, "S"], table