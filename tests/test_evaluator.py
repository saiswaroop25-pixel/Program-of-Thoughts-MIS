"""tests/test_evaluator.py — run with: python tests/test_evaluator.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.evaluator import exact_match, relaxed_match, option_match, compute_accuracy

def test_exact_match():
    assert exact_match(18.0, "18")
    assert exact_match(18,   18)
    assert not exact_match(17.0, 18)
    assert exact_match(3.14159, "3.14159")
    print("  [PASS] exact_match")

def test_relaxed_match():
    assert relaxed_match(1000000.5, 1000001.0, rel_tol=1e-3)
    assert not relaxed_match(100, 200, rel_tol=1e-3)
    print("  [PASS] relaxed_match")

def test_option_match():
    opts = ["A)1 hour", "B)2 hours", "C)3 hours", "D)4 hours", "E)5 hours"]
    assert option_match(1.0, opts) == "A"
    assert option_match(3.0, opts) == "C"
    print("  [PASS] option_match")

def test_compute_accuracy():
    results = [
        {"answer": 18,  "prediction": 18.0},
        {"answer": 42,  "prediction": 43.0},
        {"answer": 100, "prediction": 100},
    ]
    m = compute_accuracy(results)
    assert m["correct"] == 2
    assert abs(m["accuracy"] - 2/3) < 1e-6
    print("  [PASS] compute_accuracy (2/3 = 66.7%)")

if __name__ == "__main__":
    print("Running evaluator tests...")
    test_exact_match()
    test_relaxed_match()
    test_option_match()
    test_compute_accuracy()
    print("\nAll evaluator tests passed")
