"""
Utility functions for all files.
"""

from typing import Any, Optional


def check_logical_equivalence(
    original_assertions: str,
    generated_assertions: str,
    constants: Optional[str] = None
) -> dict[str, Any]:
    """
    In-process Z3 two-step equivalence check:
      1) A ⇒ B
      2) B ⇒ A.
    """
    from z3 import And, Not, Solver, Z3Exception, parse_smt2_string, sat

    orig = original_assertions.strip()
    gen = generated_assertions.strip()
    if constants:
        decls = constants.strip()
        orig = decls + f"\n{orig}"
        gen = decls + f"\n{gen}"

    # trivial cases
    if not orig and not gen:
        return {"result": True}
    if not orig or not gen:
        return {"result": False, "reason": "Empty side"}

    # try:
    #     A = parse_smt2_string(orig)
    #     B = parse_smt2_string(gen)
    # except Z3Exception as e:
    #     return {"result": False, "reason": f"Parse error: {e}"}

    try:
        A = parse_smt2_string(orig)
    except Z3Exception as e:
        return {"result": False, "reason": f"Parse error Original Constraints: {e}", "constraints": orig}
    try:
        B = parse_smt2_string(gen)
    except Z3Exception as e:
        return {"result": False, "reason": f"Parse error Generated Constraints: {e}", "constraints": gen}

    s = Solver()
    # A ⇒ B
    s.push()
    s.add(*A)
    s.add(Not(And(*B)))
    if s.check() == sat:
        return {"result": False, "reason": "A does not imply B"}
    s.pop()
    # B ⇒ A
    s.push()
    s.add(*B)
    s.add(Not(And(*A)))
    if s.check() == sat:
        return {"result": False, "reason": "B does not imply A"}
    s.pop()

    return {"result": True}



def check_logical_equivalence_v2(
    original_assertions: str,
    generated_assertions: str,
    original_constants: Optional[str] = None,
    generated_constants: Optional[str] = None
) -> dict[str, Any]:
    """
    In-process Z3 two-step equivalence check:
      1) A ⇒ B
      2) B ⇒ A.
    """
    from z3 import And, Not, Solver, Z3Exception, parse_smt2_string, sat

    orig = original_assertions.strip()
    gen = generated_assertions.strip()
    # if constants:
    #     decls = constants.strip()
    #     orig = decls + f"\n{orig}"
    #     gen = decls + f"\n{gen}"

    if generated_constants:
        gen_decls = generated_constants.strip()
        gen = gen_decls + f"\n{gen}"
    if original_constants:
        orig_decls = original_constants.strip()
        orig = orig_decls + f"\n{orig}"

    # trivial cases
    if not orig and not gen:
        return {"result": True}
    if not orig or not gen:
        return {"result": False, "reason": "Empty side"}

    # try:
    #     A = parse_smt2_string(orig)
    #     B = parse_smt2_string(gen)
    # except Z3Exception as e:
    #     return {"result": False, "reason": f"Parse error: {e}"}

    try:
        A = parse_smt2_string(orig)
    except Z3Exception as e:
        return {"result": False, "reason": f"Parse error Original Constraints: {e}", "constraints": orig}
    try:
        B = parse_smt2_string(gen)
    except Z3Exception as e:
        return {"result": False, "reason": f"Parse error Generated Constraints: {e}", "constraints": gen}

    s = Solver()
    # A ⇒ B
    s.push()
    s.add(*A)
    s.add(Not(And(*B)))
    if s.check() == sat:
        return {"result": False, "reason": "A does not imply B"}
    s.pop()
    # B ⇒ A
    s.push()
    s.add(*B)
    s.add(Not(And(*A)))
    if s.check() == sat:
        return {"result": False, "reason": "B does not imply A"}
    s.pop()

    return {"result": True}
