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

    def sentence_prob(self, sentence: str, start_ratio: float, accelerat_ratio: float):
        P = defaultdict(lambda: float("-inf"))
        table = defaultdict(None)
        status = defaultdict(bool)
        
        sentence = sentence.strip().split(" ")
        length = len(sentence)

        for i in range(1, length + 1):
            for A, w in self.grammar.unary_rules:
                if w == sentence[i - 1]:
                    P[(i, i, A)] = log(self.q.get((A, w), 0))
                    table[(i, i, A)] = [(i, i, w)]
                    table[(i, i, w)] = []


        for l in range(2, length + 1):
            for i in range(1, length + 2 - l):
                j = i + l - 1
                for k in range(i, j):
                    for A, B, C in self.grammar.binary_rules:
                        if (P.get((i, k, B), float("-inf")) != float("-inf")  
                        and P.get((k + 1, j, C), float("-inf")) != float("-inf")):
                            if A.startswith("$"):
                                if (status.get((i, k, B), False) or status.get((k + 1, j, C), False)):
                                    Prob = (P.get((i, k, B), float("-inf"))
                                        + log(self.q.get((A, B, C), 0))
                                        + P.get((k + 1, j, C), float("-inf")) 
                                        + log(accelerat_ratio))       
                                else:
                                    Prob = (P.get((i, k, B), float("-inf"))
                                       + log(self.q.get((A, B, C), 0))
                                       + P.get((k + 1, j, C), float("-inf")) 
                                       + log(start_ratio))  
                                                    
                            else:   
                                Prob = (P.get((i, k, B), float("-inf"))
                                   + log(self.q.get((A, B, C), 0))
                                   + P.get((k + 1, j, C), float("-inf")))                          
                                        
                            if Prob > P.get((i, j, A), float("-inf")):
                                P[(i, j, A)] = Prob
                                table[(i, j, A)] = [(i, k, B), (k + 1, j, C)]
                                if B.startswith("$") or C.startswith("$"):
                                    status[(i, j, A)] = True
                                else:
                                    status[(i, j, A)] = False 
                            
                                
        return P[1, length, "S"], table
    
    
    def calculate_mismatch(self, words):
        n = len(words)

        # Initialize 2D arrays for mismatch, netopen, and total mismatch
        mismatch = [[0] * n for _ in range(n)]
        netopen = [[0] * n for _ in range(n)]
        total_mismatch = [[0] * n for _ in range(n)]

        # Base case: substrings of length 1
        for i in range(n):
            if words[i] == "<":
                mismatch[i][i] = 0
                netopen[i][i] = 1  # One unmatched '<'
            elif words[i] == ">":
                mismatch[i][i] = 1  # One unmatched '>'
                netopen[i][i] = 0
            else:
                mismatch[i][i] = 0
                netopen[i][i] = 0
            total_mismatch[i][i] = mismatch[i][i] + netopen[i][i]

        # Substrings of length >= 2
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1

                # Start by copying the values from the substring words[i:j]
                mismatch[i][j] = mismatch[i][j - 1]
                netopen[i][j] = netopen[i][j - 1]

                # Incorporate the bracket at position j
                if words[j] == "<":
                    netopen[i][j] += 1  # Push one '<'
                elif words[j] == ">":
                    if netopen[i][j] > 0:
                        netopen[i][j] -= 1  # Match with a previous '<'
                    else:
                        mismatch[i][j] += 1  # Unmatched '>'

                total_mismatch[i][j] = mismatch[i][j] + netopen[i][j]

        return total_mismatch
    
    def sentence_prob__(self, sentence: str, start_ratio: float, accelerat_ratio: float, flag_ratio: float):
        # Split sentence into words and identify ignored ones
        words = sentence.strip().split(" ")
                
        filtered_sentence = []
        filtered_indices = []
        for i, w in enumerate(words):
            if w not in ("<", ">"):
                filtered_sentence.append(w)
                filtered_indices.append(i)
        
        total_mismatch = self.calculate_mismatch(words)
        
        length = len(filtered_sentence)
                
        if length == 0:
            return float(0), {}
        
        P = defaultdict(lambda: float("-inf"))
        status = defaultdict(bool)  
        table = defaultdict(None)
                    
        for i in range(1, length + 1):
            for A, w in self.grammar.unary_rules:
                if w == filtered_sentence[i - 1]:
                    P[(i, i, A)] = log(self.q.get((A, w), 0))
                    table[(i, i, A)] = [(i, i, w)]
                    table[(i, i, w)] = []

        # Binary rules (off-diagonal cells)
        for l in range(2, length + 1):  
            for i in range(1, length + 2 - l):  
                j = i + l - 1  # End index                
                for k in range(i, j):
                    for A, B, C in self.grammar.binary_rules:
                        if (P.get((i, k, B), float("-inf")) != float("-inf")  
                        and P.get((k + 1, j, C), float("-inf")) != float("-inf")):
                            if A.startswith("$"):
                                start_idx = filtered_indices[i-1]
                                end_idx   = filtered_indices[j-1]

                                sign_count = total_mismatch[start_idx][end_idx]
                                                                
                                if status.get((i, k, B), False) or status.get((k + 1, j, C), False):
                                    Prob = (P.get((i, k, B), float("-inf"))
                                           + log(self.q.get((A, B, C), 0))
                                           + P.get((k + 1, j, C), float("-inf")) 
                                           + log(pow(flag_ratio, sign_count)) 
                                           + log(accelerat_ratio))
                                else:                                                                
                                    Prob = (P.get((i, k, B), float("-inf"))
                                           + log(self.q.get((A, B, C), 0))
                                           + P.get((k + 1, j, C), float("-inf")) 
                                           + log(pow(flag_ratio, sign_count))
                                           + log(start_ratio))
                                    
                            else:
                                Prob = (P.get((i, k, B), float("-inf"))
                                    + log(self.q.get((A, B, C), 0))
                                    + P.get((k + 1, j, C), float("-inf")))
                                     
                            if Prob > P.get((i, j, A), float("-inf")):
                                P[(i, j, A)] = Prob
                                table[(i, j, A)] = [(i, k, B), (k + 1, j, C)]
                                if B.startswith("$") or C.startswith("$"):
                                    status[(i, j, A)] = True
                                else:
                                    status[(i, j, A)] = False 
                                    
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