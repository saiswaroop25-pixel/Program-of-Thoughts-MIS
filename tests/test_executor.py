"""tests/test_executor.py — run with: python tests/test_executor.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.executor import execute_program

def test_basic_arithmetic():
    code = "total=16\neaten=3\nbaked=4\nsold=total-eaten-baked\nans=sold*2"
    ans, err = execute_program(code)
    assert ans == 18, f"Expected 18, got {ans}"
    print("  [PASS] basic arithmetic")

def test_sympy_solve():
    code = "from sympy import Symbol, solve\nx=Symbol('x',positive=True)\nans=float(solve(x**2-25,x)[0])"
    ans, err = execute_program(code)
    assert abs(ans - 5.0) < 1e-6, f"Expected 5.0, got {ans}"
    print("  [PASS] sympy solve")

def test_loop():
    code = "seq=[0,1]\nfor i in range(2,10):\n    seq.append(seq[-1]+seq[-2])\nans=seq[-1]"
    ans, err = execute_program(code)
    assert ans == 34, f"Expected 34, got {ans}"
    print("  [PASS] loop (fibonacci)")

def test_dangerous_blocked():
    code = "import os\nans=os.listdir('/')"
    ans, err = execute_program(code)
    assert ans is None, "Dangerous import should be blocked"
    print("  [PASS] dangerous import blocked")

def test_timeout():
    code = "ans=0\nwhile True:\n    ans+=1"
    ans, err = execute_program(code, timeout=2)
    assert ans is None and err is not None
    print("  [PASS] infinite loop timed out")

if __name__ == "__main__":
    print("Running executor tests...")
    test_basic_arithmetic()
    test_sympy_solve()
    test_loop()
    test_dangerous_blocked()
    test_timeout()
    print("\nAll executor tests passed")
