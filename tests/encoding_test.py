import unittest
from cbmp_cnf import pairwise_amo
from pysat.card import EncType, CardEnc
from pysat.formula import CNF

test_list = [1, 2, 3, 4, 5, 6, 7]

class EncodingTest(unittest.TestCase):
    def test_encoding_correctness(self):
        custom_cnf = CNF()
        ref_cnf = CardEnc.atmost(lits = test_list, encoding=EncType.pairwise)
        pairwise_amo(test_list, custom_cnf)
        self.assertCountEqual(
            {frozenset(c) for c in ref_cnf.clauses},
            {frozenset(c) for c in custom_cnf.clauses},
        )
        