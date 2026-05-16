"""tests/test_extensions.py — run with: python tests/test_extensions.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions.math_verifier       import verify_answer
from extensions.complexity_analyzer import analyse, route_method
from extensions.self_consistency    import _majority_vote, _count_votes

def test_math_verifier_correct():
    code = "total=16\neaten=3\nbaked=4\nsold=total-eaten-baked\nprice=2\nans=sold*price"
    v = verify_answer(code, predicted=18.0)
    assert v["verified"] == True, f"Expected True, got {v}"
    print("  [PASS] verifier accepts correct answer")

def test_math_verifier_wrong():
    code = "total=16\neaten=3\nbaked=4\nsold=total-eaten-baked\nprice=2\nans=sold*price"
    v = verify_answer(code, predicted=99.0)
    assert v["verified"] == False, f"Expected False, got {v}"
    print("  [PASS] verifier rejects wrong answer")

def test_complexity_iterative():
    q = "What is the 50th Fibonacci number if each number equals sum of the two preceding?"
    m, a = route_method(q)
    assert a.category == "iterative"
    assert m == "pot_only"
    print(f"  [PASS] iterative -> {m}")

def test_complexity_equation():
    q = "Find the rate of interest if compound interest minus simple interest for 2 years is Rs.72"
    m, a = route_method(q)
    assert m in ("pot_verify", "pot_only")
    print(f"  [PASS] equation -> {m} ({a.category})")

def test_complexity_geometry():
    q = "Find the area of a triangle with base 5cm and height 3cm."
    m, a = route_method(q)
    assert m == "cot_only"
    print(f"  [PASS] geometry -> {m}")

def test_majority_vote():
    answers = [18.0, 18.0, 17.0, 18.0, 19.0]
    mv = _majority_vote(answers)
    assert mv == 18.0, f"Expected 18.0, got {mv}"
    print(f"  [PASS] majority vote: {answers} -> {mv}")

def test_vote_counts():
    answers = [18.0, 18.0, 17.0, 18.0]
    counts = _count_votes(answers)
    assert counts.get("18.0") == 3
    print(f"  [PASS] vote counts: {counts}")

if __name__ == "__main__":
    print("Running extension tests...")
    test_math_verifier_correct()
    test_math_verifier_wrong()
    test_complexity_iterative()
    test_complexity_equation()
    test_complexity_geometry()
    test_majority_vote()
    test_vote_counts()
    print("\nAll extension tests passed")
