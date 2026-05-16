"""Safe execution for LLM-generated Program-of-Thoughts Python code."""

from __future__ import annotations

import math
import base64
import json
import re
import subprocess
import sys
import traceback
from fractions import Fraction
from typing import Any, Optional, Tuple

import sympy


SAFE_MODULES = {
    "math": math,
    "sympy": sympy,
    "fractions": __import__("fractions"),
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root_name = name.split(".", 1)[0]
    if level != 0 or root_name not in SAFE_MODULES:
        raise ImportError(f"Import of {name!r} is blocked by the PoT executor")
    if root_name == "sympy" and "solve_it" in fromlist and "solve_it" in SAFE_GLOBALS:
        setattr(SAFE_MODULES["sympy"], "solve_it", SAFE_GLOBALS["solve_it"])
    return __import__(name, globals, locals, fromlist, level)


SAFE_GLOBALS = {
    "__builtins__": {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
        "reversed": reversed,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "print": print,
        "isinstance": isinstance,
        "type": type,
        "True": True,
        "False": False,
        "None": None,
        "__import__": _safe_import,
    },
    "math": math,
    "sympy": sympy,
    "Fraction": Fraction,
}

for _name in [
    "Symbol",
    "symbols",
    "solve",
    "simplify",
    "sqrt",
    "pi",
    "Rational",
    "oo",
    "log",
    "exp",
    "cos",
    "sin",
    "tan",
    "floor",
    "ceiling",
    "factorial",
    "binomial",
    "Eq",
    "Piecewise",
    "Sum",
    "Product",
    "Integral",
    "diff",
]:
    SAFE_GLOBALS[_name] = getattr(sympy, _name, None)

try:
    import numpy as np
except ImportError:
    np = None
else:
    SAFE_MODULES["numpy"] = np
    SAFE_GLOBALS["np"] = np
    SAFE_GLOBALS["numpy"] = np


def solve_it(equations, variables):
    """Small SymPy helper used by the original PoT prompts."""
    if not isinstance(equations, (list, tuple)):
        equations = [equations]
    if not isinstance(variables, (list, tuple)):
        variables = [variables]

    solutions = sympy.solve(equations, variables, dict=True)
    if not solutions:
        return {}
    for sol in solutions:
        if all(v.is_real for v in sol.values()):
            return {str(k): float(v) for k, v in sol.items()}
    return {str(k): complex(v) for k, v in solutions[0].items()}


SAFE_GLOBALS["solve_it"] = solve_it


def execute_program(code: str, timeout: int = 10) -> Tuple[Optional[Any], Optional[str]]:
    """Execute generated code and return (ans, error)."""
    code = _clean_code(code)
    if not code.strip():
        return None, "Empty code"

    payload = base64.b64encode(code.encode("utf-8")).decode("ascii")
    command = [sys.executable, __file__, "--execute-worker", payload]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "TimeoutError: execution exceeded time limit"

    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("__POT_RESULT__"):
            result = json.loads(line.removeprefix("__POT_RESULT__"))
            if result["ok"]:
                return result["answer"], None
            return None, result["error"]

    error = completed.stderr.strip() or completed.stdout.strip()
    return None, error or "Execution failed without returning a result"


def _execute_worker(code: str) -> dict:
    local_vars: dict = {}
    global_vars = dict(SAFE_GLOBALS)

    try:
        exec(code, global_vars, local_vars)  # noqa: S102
        ans = local_vars.get("ans", global_vars.get("ans", None))
        return {"ok": True, "answer": _coerce_answer(ans)}
    except Exception:  # noqa: BLE001
        return {"ok": False, "error": traceback.format_exc(limit=3)}


def _clean_code(code: str) -> str:
    """Strip markdown fences and leading/trailing whitespace."""
    code = code.strip()
    code = re.sub(r"^```(?:python)?\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\s*```$", "", code)
    return code.strip()


def _coerce_answer(ans: Any) -> Any:
    """Convert SymPy and NumPy values to plain Python values."""
    if ans is None:
        return None
    try:
        if hasattr(ans, "is_number") and ans.is_number:
            return float(ans.evalf())
    except Exception:
        pass
    if np is not None and isinstance(ans, np.generic):
        return ans.item()
    if isinstance(ans, (tuple, list)):
        return [_coerce_answer(a) for a in ans]
    return ans


def _main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--execute-worker":
        code = base64.b64decode(sys.argv[2]).decode("utf-8")
        result = _execute_worker(code)
        print("__POT_RESULT__" + json.dumps(result, default=str))


if __name__ == "__main__":
    _main()
