import pickle
from collections import defaultdict

from argtyped import Arguments
from typing import List, Optional, NamedTuple, Tuple, Iterator, Callable, Dict, TypeVar
import numpy as np
from termcolor import colored


class Args(Arguments):
    test_file: str = "test_output.pkl"


class Frac:
    def __init__(self, numerator: int = 0, denominator: int = 0):
        self.numerator = numerator
        self.denominator = denominator

    def add(self, examples: List[bool]) -> None:
        self.numerator += sum(examples)
        self.denominator += len(examples)

    def __float__(self) -> float:
        return self.numerator / self.denominator

    def __str__(self) -> str:
        return f"{self.numerator} / {self.denominator}"


class ConfusionMat:
    def __init__(self):
        self.matrix = np.zeros((2, 2), dtype=np.int)

    @property
    def accuracy(self) -> float:
        return (self.matrix[0, 0] + self.matrix[1, 1]) / np.sum(self.matrix)

    @property
    def true_positive(self) -> int:
        return self.matrix[1, 1]

    @property
    def false_positive(self) -> int:
        return self.matrix[0, 1]

    @property
    def true_negative(self) -> int:
        return self.matrix[0, 0]

    @property
    def false_negative(self) -> int:
        return self.matrix[1, 0]

    @property
    def precision(self) -> Frac:
        return Frac(self.true_positive, self.true_positive + self.false_positive)

    @property
    def recall(self) -> Frac:
        return Frac(self.true_positive, self.true_positive + self.false_negative)

    @property
    def f1(self) -> float:
        precision = float(self.precision)
        recall = float(self.recall)
        return 2 * precision * recall / (precision + recall)

    def add(self, *, gold: Optional[List[bool]] = None, pred: List[bool]) -> None:
        if gold is None:
            gold = [True] * len(pred)
        for g, p in zip(gold, pred):
            self.matrix[int(g), int(p)] += 1


T = TypeVar('T')


class TypeSignature(NamedTuple):
    type: List[str]
    pointer_layer: int  # `char **p` => 2 layers


class FuncSignature(NamedTuple):
    ret_type: TypeSignature
    name: str
    args: List[Tuple[TypeSignature, str]]  # [(type, name)]


class Evaluator:
    def __init__(self):
        self.stat_unparsable = Frac()  # code that is unparsable
        self.stat_fn_name = Frac()  # correct function name
        self.stat_fn_ret_type = Frac()  # correct function return type
        self.stat_arg_name = Frac()  # correct argument names (w.r.t arguments in target)
        self.stat_arg_type = Frac()  # correct argument types
        self.stat_arg_missing = Frac()  # missing arguments
        self.stat_redundant_args = 0  # extra/duplicate arguments
        self.stat_pointer = ConfusionMat()  # correct type changes from non-pointer to pointer

    @classmethod
    def _parse_func(cls, code: List[str], syntax_correct: bool = True) -> FuncSignature:
        braces_map = {
            "{": ("{}", 1),
            "}": ("{}", -1),
            "(": ("()", 1),
            ")": ("()", -1),
            "[": ("[]", 1),
            "]": ("[]", -1),
        }

        def find_next(token: str, start: int = 0) -> int:
            # Find first occurrence of `token` in `[start, len(code))`.
            return next(idx for idx in range(start, len(code)) if code[idx] == token)

        def find_prev(token: str, end: int) -> int:
            # Find last occurrence of `token` in `[0, end]`
            if end < 0: end += len(code)
            return next(idx for idx in range(end, -1, -1) if code[idx] == token)

        def _check_balance(indices: Iterator[int], callback: Callable[[int, bool, Dict[str, int]], T]) -> T:
            # callback: (idx, is_balanced, balance) -> ret?
            balance = defaultdict(int)
            for idx in indices:
                if code[idx] in braces_map:
                    kind, delta = braces_map[code[idx]]
                    balance[kind] += delta
                is_balance = all(v == 0 for v in balance.values())
                ret = callback(idx, is_balance, balance)
                if ret is not None:
                    return ret

        def _find_match(indices: Iterator[int], token: str) -> int:
            ret = _check_balance(indices, lambda idx, is_bal, _: idx if is_bal else None)
            assert ret is not None and code[ret] == token
            return ret

        def find_match_left(braces: str, r_pos: int) -> int:
            # Find matching left brace given right position.
            if r_pos < 0: r_pos += len(code)
            assert len(braces) == 2 and code[r_pos] == braces[1]
            return _find_match(range(r_pos, -1, -1), braces[0])

        def find_match_right(braces: str, l_pos: int) -> int:
            # Find matching right brace given left position.
            assert len(braces) == 2 and code[l_pos] == braces[0]
            return _find_match(range(l_pos, len(code)), braces[1])

        def find_token_within(token: str, l: int, r: int) -> List[int]:
            # Find occurrences of `token` within `[l, r]` that are not enclosed by braces.
            indices = []
            _check_balance(
                range(l, r + 1), lambda idx, is_bal, _:
                indices.append(idx) if code[idx] == token and is_bal else None)
            return indices

        def parse_vardef(l: int, r: int) -> Tuple[TypeSignature, str]:
            # Parse a variable definition within `[l, r]` and return the type and variable name.
            # The variable name is (usually?) the leftmost identifier within the outermost '(' group.
            # If no '('s exist, then the variable name is the rightmost identifier (`type var;`).
            # The type is the rest after deleting the identifier.
            if code[l:(r + 1)] == ["..."]:
                # varargs
                return TypeSignature(["..."], False), "..."

            def callback(idx: int, _, balance: Dict[str, int]) -> Optional[int]:
                if balance['[]'] == 0 and balance['{}'] == 0 and code[idx].isidentifier():
                    return idx

            lparen_pos = _check_balance(
                range(l, r + 1), lambda idx, is_bal, _:
                idx if is_bal and code[idx] == '(' else None)
            if lparen_pos is None:
                # Find rightmost identifier.
                index = _check_balance(range(r, l - 1, -1), callback)
            else:
                rparen_pos = find_match_right("()", lparen_pos)
                index = _check_balance(range(lparen_pos, rparen_pos + 1), callback)
            assert index is not None
            name = code[index]

            new_type = []
            _check_balance(
                range(l, r + 1), lambda idx, _, balance:
                new_type.append(code[idx]) if idx != index and balance['[]'] == 0 else None)
            ptr_level = new_type.count("*")
            return TypeSignature(new_type, ptr_level), name

        if syntax_correct:
            assert code[-1] == "}"
            # Find matching '{' for the final '}'
            # These enclose the function body and everything before is the signature.
            body_lbrace_pos = find_match_left("{}", -1)
        else:
            # The generated code might not be syntactically correct.
            body_lbrace_pos = find_next("{")
        # Find the first ')' before.
        arg_rparen_pos = find_prev(")", body_lbrace_pos)
        arg_lparen_pos = find_match_left("()", arg_rparen_pos)

        # Function name and return type is represented as a variable declaration before '('.
        ret_type, func_name = parse_vardef(0, arg_lparen_pos - 1)
        # Arguments are separated by ',' within arguments '()'.
        if arg_lparen_pos + 1 == arg_rparen_pos or code[arg_lparen_pos + 1] == "void":
            args = []
        else:
            if arg_rparen_pos == body_lbrace_pos - 1:
                comma_pos = find_token_within(",", arg_lparen_pos + 1, arg_rparen_pos - 1)
                arg_boundaries = [arg_lparen_pos] + comma_pos + [arg_rparen_pos]
            else:
                # This might be the legacy K&R-style definition.
                semicolon_pos = find_token_within(";", arg_rparen_pos + 1, body_lbrace_pos - 1)
                arg_boundaries = [arg_rparen_pos] + semicolon_pos
            args = [parse_vardef(l + 1, r - 1) for l, r in zip(arg_boundaries, arg_boundaries[1:])]

        return FuncSignature(ret_type, func_name, args)

    @classmethod
    def _split_code(cls, code: str, syntax_correct: bool = True) -> List[str]:
        # Split code considering string literals.
        string_start: Optional[int] = None
        tokens = []
        for idx, token in enumerate(code.split()):
            if string_start is not None:
                if token[-1] == '"' and (len(token) == 1 or token[-2] != '\\'):
                    tokens.append(" ".join(code[string_start:(idx + 1)]))
                    string_start = None
            elif token[0] == '"' and (len(token) == 1 or token[-1] != '"'):
                string_start = idx
            else:
                tokens.append(token)
        if string_start is not None:
            assert not syntax_correct
            tokens.append(" ".join(code[string_start:]))
        return tokens

    @classmethod
    def _compare_type(cls, a: TypeSignature, b: TypeSignature, strict: bool = False) -> bool:
        # If not `strict`, discard qualifiers.
        qualifiers = {"static", "const", "volatile", "restrict", "signed"}
        typ_a = a.type.copy()
        typ_b = b.type.copy()
        if not strict:
            typ_a = [x for x in typ_a if x not in qualifiers]
            typ_b = [x for x in typ_b if x not in qualifiers]
        return typ_a == typ_b

    def add(self, src: str, tgt: str, hyp: str) -> None:
        src_func_sig = self._parse_func(self._split_code(src))
        # if src_func_sig.name == "tree_add":
        #     breakpoint()
        tgt_func_sig = self._parse_func(self._split_code(tgt))
        try:
            hyp_func_sig = self._parse_func(self._split_code(hyp, syntax_correct=False), syntax_correct=False)
        except:
            print(colored("Malformed output:", "red"), hyp)
            self.stat_unparsable.add([True])
            hyp_func_sig = FuncSignature(TypeSignature(["void"], False), "", [])

        # assert src_func_sig.name == tgt_func_sig.name
        if src_func_sig.name != tgt_func_sig.name:
            print(colored(src_func_sig, "green"))
            print(self._split_code(src))
            print(colored(tgt_func_sig, "green"))
            print(self._split_code(tgt))
            print(colored(hyp_func_sig, "green"))
            print(self._split_code(hyp))
            return

        self.stat_unparsable.add([False])
        self.stat_fn_name.add([tgt_func_sig.name == hyp_func_sig.name])
        self.stat_fn_ret_type.add([self._compare_type(tgt_func_sig.ret_type, hyp_func_sig.ret_type)])
        missing = []
        pointer_check = [(src_func_sig.ret_type, tgt_func_sig.ret_type, hyp_func_sig.ret_type)]
        for tgt_arg_type, tgt_arg_name in tgt_func_sig.args:
            idx = next((idx for idx, (_, name) in enumerate(hyp_func_sig.args) if name == tgt_arg_name), None)
            missing.append(idx is None)
            if idx is not None:
                hyp_arg_typ, _ = hyp_func_sig.args[idx]
                self.stat_arg_type.add([self._compare_type(tgt_arg_type, hyp_arg_typ)])
                src_arg_typ = next((typ for typ, name in src_func_sig.args if name == tgt_arg_name), None)
                if src_arg_typ is not None:
                    pointer_check.append((src_arg_typ, tgt_arg_type, hyp_arg_typ))
                del hyp_func_sig.args[idx]
        self.stat_arg_missing.add(missing)
        self.stat_arg_name.add([not x for x in missing])
        self.stat_redundant_args += len(hyp_func_sig.args)
        for src_typ, tgt_typ, hyp_typ in pointer_check:
            if not src_typ.pointer_layer:
                self.stat_pointer.add(gold=[tgt_typ.pointer_layer > 0], pred=[hyp_typ.pointer_layer > 0])

    def print_summary(self) -> None:
        print(colored("Unparsable:", "red"), self.stat_unparsable)
        print(colored("Correct func names:", "green"), self.stat_fn_name)
        print(colored("Correct return types:", "green"), self.stat_fn_ret_type)
        print(colored("Correct argument names:", "green"), self.stat_arg_name)
        print(colored("Correct argument types:", "green"), self.stat_arg_type)
        print(colored("Missing arguments:", "red"), self.stat_arg_missing)
        print(colored("Redundant arguments:", "red"), self.stat_redundant_args)
        print(colored("Pointer conversion:", "green"),
              f"precision: {self.stat_pointer.precision}, recall {self.stat_pointer.recall}")


def main():
    args = Args()
    with open(args.test_file, "rb") as f:
        src_data, tgt_data, hyp_data, overlap_scores = pickle.load(f)

    evaluator = Evaluator()
    # for src, tgt, hyp in zip(src_data, tgt_data, hyp_data):
    #     evaluator.add(src, tgt, hyp)
    for src, tgt, hyp in zip(src_data, tgt_data, src_data):
        evaluator.add(src, tgt, hyp)

    evaluator.print_summary()


if __name__ == '__main__':
    main()
