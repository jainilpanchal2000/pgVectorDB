
import unittest
import sys
from pathlib import Path

# Add parent directory to path to import src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.metrics import RAGEvaluator

class TestRAGMetrics(unittest.TestCase):
    def setUp(self):
        self.k = 5
        self.evaluator = RAGEvaluator(k=self.k)

    def test_precision_at_k(self):
        # Case 1: 2 relevant in top 5 -> Precision = 2/5 = 0.4
        retrieved = ["r1", "i1", "r2", "i2", "i3"]
        relevant = ["r1", "r2", "r3"]
        metrics = self.evaluator.evaluate_single_query(retrieved, relevant)
        self.assertAlmostEqual(metrics['precision'], 0.4)

        # Case 2: 0 relevant -> 0.0
        retrieved = ["i1", "i2", "i3", "i4", "i5"]
        metrics = self.evaluator.evaluate_single_query(retrieved, relevant)
        self.assertEqual(metrics['precision'], 0.0)
        
        # Case 3: All relevant -> 1.0
        retrieved = ["r1", "r2", "r3"] # Less than k, but all relevant
        relevant = ["r1", "r2", "r3"]
        metrics = self.evaluator.evaluate_single_query(retrieved, relevant)
        # Precision is retrieved_intersect_relevant / k
        # Here retrieved has 3 items, all relevant. 
        # Wait, the code says: precision_at_k = true_positives / self.k
        # So 3 / 5 = 0.6. This is strict P@K.
        # Let's verify if the code works this way.
        self.assertAlmostEqual(metrics['precision'], 0.6)

    def test_recall_at_k(self):
        # Case 1: 2 relevant retrieved out of 4 total relevant -> Recall = 2/4 = 0.5
        retrieved = ["r1", "i1", "r2"]
        relevant = ["r1", "r2", "r3", "r4"]
        metrics = self.evaluator.evaluate_single_query(retrieved, relevant)
        self.assertAlmostEqual(metrics['recall'], 0.5)

    def test_mrr(self):
        relevant = ["r1", "r2"]
        
        # Case 1: First relevant at index 0 (rank 1) -> MRR = 1/1 = 1.0
        retrieved = ["r1", "i1", "i2"]
        metrics = self.evaluator.evaluate_single_query(retrieved, relevant)
        self.assertEqual(metrics['reciprocal_rank'], 1.0)
        
        # Case 2: First relevant at index 2 (rank 3) -> MRR = 1/3
        retrieved = ["i1", "i2", "r1"]
        metrics = self.evaluator.evaluate_single_query(retrieved, relevant)
        self.assertAlmostEqual(metrics['reciprocal_rank'], 1.0/3.0)
        
        # Case 3: No relevant -> MRR = 0
        retrieved = ["i1", "i2", "i3"]
        metrics = self.evaluator.evaluate_single_query(retrieved, relevant)
        self.assertEqual(metrics['reciprocal_rank'], 0.0)

    def test_map(self):
        # Average Precision = (Sum of P@k for each relevant item) / Total Relevant
        relevant = ["r1", "r2", "r3"]
        
        # Retrieval: [r1(rel), i1, r2(rel), i2, r3(rel)]
        retrieved = ["r1", "i1", "r2", "i2", "r3"]
        
        # P@1 (r1): 1/1 = 1.0
        # P@3 (r2): 2/3 = 0.666...
        # P@5 (r3): 3/5 = 0.6
        # AP = (1.0 + 0.666 + 0.6) / 3 = 2.266 / 3 = 0.755...
        
        metrics = self.evaluator.evaluate_single_query(retrieved, relevant)
        self.assertAlmostEqual(metrics['average_precision'], (1.0 + 2/3 + 3/5) / 3)

    def test_ndcg(self):
        import numpy as np
        # IDCG for 3 relevant items:
        # 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.0 + 0.6309 + 0.5 = 2.1309
        
        relevant = ["r1", "r2", "r3"]
        
        # Retrieved: [r1, r2, r3] -> Perfect order
        retrieved = ["r1", "r2", "r3"]
        metrics = self.evaluator.evaluate_single_query(retrieved, relevant)
        self.assertAlmostEqual(metrics['ndcg'], 1.0)
        
        # Retrieved: [i1, r1, i2]
        # DCG = 0 + 1/log2(3) + 0 = 0.6309
        # IDCG (since we have 3 relevant, ideal is still top 3 filled)
        # IDCG @ k=5 (or length of retrieved?) Code uses min(len(relevant), k)
        # So IDCG is for 3 items.
        # NDCG = 0.6309 / 2.1309
        retrieved = ["i1", "r1", "i2"]
        metrics = self.evaluator.evaluate_single_query(retrieved, relevant)
        
        expected_dcg = 1.0 / np.log2(3)
        expected_idcg = 1.0 + 1.0/np.log2(3) + 1.0/np.log2(4)
        self.assertAlmostEqual(metrics['ndcg'], expected_dcg / expected_idcg)

if __name__ == '__main__':
    unittest.main()
