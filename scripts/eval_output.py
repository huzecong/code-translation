import itertools
import os
import pickle
import re
import string
import subprocess
from collections import Counter, OrderedDict, defaultdict
from typing import Callable, Dict, Generic, Iterable, Iterator, List, NamedTuple, Optional, Tuple, TypeVar

import flutes
import numpy as np
import texar.torch as tx
from argtyped import Arguments
from mypy_extensions import TypedDict
from termcolor import colored
from tqdm import trange


class Args(Arguments):
    test_file: str = "test_output.pkl"
    output_dir: str = "."


class Frac:
    def __init__(self, numerator: int = 0, denominator: int = 0):
        self.numerator = numerator
        self.denominator = denominator

    def add(self, examples: Iterable[bool]) -> None:
        for ex in examples:
            self.numerator += ex
            self.denominator += 1

    def __float__(self) -> float:
        return self.numerator / self.denominator

    def __str__(self) -> str:
        return f"{self.numerator} / {self.denominator}"


T = TypeVar('T')
R = TypeVar('R')


class CategoryCounter(Generic[T]):
    def __init__(self):
        self.counter: 'Counter[T]' = Counter()

    def add(self, examples: Iterable[T]) -> None:
        self.counter.update(examples)

    def __str__(self):
        return self.to_string()

    def group_by(self, group_fn: Callable[[T], R]) -> Dict[R, List[Tuple[T, int]]]:
        groups: Dict[R, List[Tuple[T, int]]] = defaultdict(list)
        for k, v in self.counter.items():
            groups[group_fn(k)].append((k, v))
        return groups

    def to_string(self, group_fn: Optional[Callable[[T], R]] = None) -> str:
        if group_fn is not None:
            groups = self.group_by(group_fn)
        else:
            groups = {None: list(self.counter.items())}
        strings = []
        for key, group in sorted(groups.items()):
            total = sum(v for _, v in group)
            vals = ", ".join(f"{k}: {v} / {total}" for k, v in sorted(group))
            strings.append((key + ": " if key is not None else "") + vals)
        return "\n".join(strings)


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

    def add(self, *, gold: Optional[Iterable[bool]] = None, pred: Iterable[bool]) -> None:
        if gold is None:
            gold = itertools.repeat(True)
        for g, p in zip(gold, pred):
            self.matrix[int(g), int(p)] += 1


def sign(x: int) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


class TypeSignature(NamedTuple):
    type: List[str]
    pointer_layer: int  # `char **p` => 2 layers


class FuncSignature(NamedTuple):
    ret_type: TypeSignature
    name: str
    args: List[Tuple[TypeSignature, str]]  # [(type, name)]


class Markdown:
    @staticmethod
    def code(s: str) -> str:
        return f"`{s}`"

    @staticmethod
    def code_block(s: str) -> str:
        return "```c\n" + s.strip() + "\n```"

    @classmethod
    def bold(cls, s: str) -> str:
        return f"**{s}**"

    @classmethod
    def underline(cls, s: str) -> str:
        return f"<u>{s}</u>"

    @classmethod
    def list(cls, lines: List[str], indent: int = 0) -> str:
        indent_str = " " * indent
        return "\n".join(indent_str + "- " + line for line in lines)

    @classmethod
    def bool(cls, val: bool, s: str) -> str:
        return f'<div class="{"correct" if val else "wrong"}">{s}</div>'

    @classmethod
    def to_id(cls, s: str) -> str:
        valid_chars = string.ascii_lowercase + string.digits + "_-"
        return "".join(filter(lambda x: x in valid_chars, s.lower().replace("_", "-").replace(" ", "-")))

    @classmethod
    def collapse_section(cls, title: str, section_id: str, contents: List[str]) -> List[str]:
        return ([f'<h4 id="{section_id}">'
                 f'  <div class="collapse-trigger" collapse="{section_id}" collapsed>{title}</div>'
                 f'</h4>',
                 f'<div class="collapse" collapse="{section_id}" collapsed>'] +
                contents +
                ["</div>", "<hr />"])

    @classmethod
    def strip_colored(cls, s: str) -> str:
        # Strip ANSI escapes.
        s = re.sub(r"\033\[\d+m", "", s)
        return s

    @classmethod
    def _safe_len(cls, s: str) -> int:
        return len(cls.strip_colored(s))

    @classmethod
    def replace_colored(cls, s: str, trans_fn: Callable[[str], str]) -> str:
        regex = re.compile(r"(\033\[\d+m)(.*?)(\033\[0m)")
        m = regex.match(s)
        if m is None:
            return trans_fn(s)
        return m.group(1) + trans_fn(m.group(2)) + m.group(3)

    @classmethod
    def table(cls, table: List[List[str]], align: List[str]) -> str:
        rows = []
        width = [max(cls._safe_len(row[idx]) for row in table) for idx in range(len(table[0]))]
        for table_row in table:
            row = []
            for idx, value in enumerate(table_row):
                if align[idx] == "right":
                    value = cls.replace_colored(value, lambda s: s.rjust(width[idx]))
                elif align[idx] == "center":
                    value = cls.replace_colored(value, lambda s: s.center(width[idx]))
                else:
                    value = cls.replace_colored(value, lambda s: s.ljust(width[idx]))
                row.append(value)
            rows.append(" | ".join(row))
        rules = []
        for idx in range(len(width)):
            if align[idx] == "right":
                rule = "-" * (width[idx] - 1) + ":"
            elif align[idx] == "center":
                rule = ":" + "-" * (width[idx] - 2) + ":"
            else:
                rule = "-" * width[idx]
            rules.append(rule)
        lines = [rows[0], " | ".join(rules)] + rows[1:]
        return "\n".join("| " + line + " |" for line in lines)

    @classmethod
    def link(cls, text: str, url: str) -> str:
        return f"[{text}]({url})"


class Stats:
    KEYS = ["unparsable",
            "arg_name",
            "arg_type",
            "arg_type_strict",
            "func_name",
            "pointer_conversion",
            "ret_type",
            "ret_type_strict"]

    def __init__(self):
        self.unparsable = Frac()  # code that is unparsable
        self.fn_name = Frac()  # correct function name
        self.fn_ret_type = Frac()  # correct function return type
        self.fn_ret_type_strict = Frac()  # correct function return type
        self.arg_name = Frac()  # correct argument names (w.r.t arguments in target)
        self.arg_type = Frac()  # correct argument types (ignoring cv-qualifiers)
        self.arg_type_strict = Frac()  # correct argument types
        self.arg_missing = Frac()  # missing arguments
        self.redundant_args = 0  # extra/duplicate arguments
        self.pointer = ConfusionMat()  # correct type changes from non-pointer to pointer

        self.improving = CategoryCounter()  # examples that improved compared to
        self.arg_name_kind = CategoryCounter()
        self.arg_type_kind = CategoryCounter()
        self.deteriorated_examples = {key: [] for key in self.KEYS}


class Evaluator:
    DECOMPILED_NAME = "Decompiled"

    def __init__(self, names: List[str], export: Optional[str] = None):
        self.stats = OrderedDict((name, Stats()) for name in [self.DECOMPILED_NAME] + names)
        self.names = names
        self.name_ids = OrderedDict((name, Markdown.to_id(name)) for name in names)

        self.references = []
        self.hypotheses = {name: [] for name in [self.DECOMPILED_NAME] + names}

        self.export_path = export
        if export is not None:
            self.index: int = 0
            self.export_sections: List[str] = []

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
            if end < 0:
                end += len(code)
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
            if r_pos < 0:
                r_pos += len(code)
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
                range(l, r + 1), lambda idx, _, balance:
                idx if balance['[]'] == 0 and balance['{}'] == 0 and code[idx] == '(' else None)
            if lparen_pos is None:
                # Find rightmost identifier.
                index = _check_balance(range(r, l - 1, -1), callback)
            else:
                rparen_pos = find_match_right("()", lparen_pos)
                index = _check_balance(range(lparen_pos, rparen_pos + 1), callback)
            assert index is not None
            name = code[index]

            ptr_level = code[l:index].count("*")
            new_type = []
            _check_balance(
                range(l, r + 1), lambda idx, _, balance:
                new_type.append(code[idx]) if idx != index and (code[idx] in "[]" or balance['[]'] == 0) else None)
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

        if ret_type == "":
            ret_type = "void"
        return FuncSignature(ret_type, func_name, args)

    @classmethod
    def _split_code(cls, code: str, syntax_correct: bool = True) -> List[str]:
        # Split code considering string literals.
        string_start: Optional[int] = None
        tokens = []
        # Replace `<unk>` with `__unk__` so it can still be part of an identifier.
        split_code = code.replace("<unk>", "__unk__").split()
        for idx, token in enumerate(split_code):
            if string_start is not None:
                if token[-1] == '"' and (len(token) == 1 or token[-2] != '\\'):
                    tokens.append(" ".join(split_code[string_start:(idx + 1)]))
                    string_start = None
            elif token[0] == '"' and (len(token) == 1 or token[-1] != '"'):
                string_start = idx
            else:
                tokens.append(token)
        if string_start is not None:
            assert not syntax_correct
            tokens.append(" ".join(split_code[string_start:]))
        return tokens

    @classmethod
    def _compare_type(cls, a: TypeSignature, b: TypeSignature,
                      cv_qualifiers: bool = False, storage_qualifiers: bool = False) -> bool:
        # If not `strict`, discard qualifiers.
        qualifiers = {"signed"}
        if not cv_qualifiers:
            qualifiers.update(["const", "volatile"])
        if not storage_qualifiers:
            qualifiers.update(["static", "auto", "register", "extern", "restrict"])
        typ_a = [x for x in a.type if x not in qualifiers]
        typ_b = [x for x in b.type if x not in qualifiers]
        return typ_a == typ_b

    class EvalOutput(NamedTuple):
        func_name: bool  # is_correct
        ret_type: bool  # is_correct
        ret_type_strict: bool  # is_correct
        missing_args: Dict[str, bool]  # (name) -> is_missing
        redundant_args: List[str]  # [name]
        arg_types: Dict[str, bool]  # (name) -> is_correct
        arg_types_strict: Dict[str, bool]  # (name) -> is_correct
        pointer_conversion: Dict[str, Tuple[bool, bool]]  # (name) -> (should_convert, did_convert)

    def _evaluate_signatures(self, src: FuncSignature, tgt: FuncSignature, hyp: FuncSignature,
                             stats: Optional[Stats] = None):
        correct_func_name = tgt.name == hyp.name
        correct_ret_type = self._compare_type(tgt.ret_type, hyp.ret_type)
        correct_ret_type_strict = self._compare_type(tgt.ret_type, hyp.ret_type, cv_qualifiers=True)
        missing: Dict[str, bool] = {}
        correct_arg_types: Dict[str, bool] = {}
        correct_arg_types_strict: Dict[str, bool] = {}
        pointer_check_types = {"": (src.ret_type, tgt.ret_type, hyp.ret_type)}
        hyp_args = hyp.args.copy()
        for tgt_arg_type, arg_name in tgt.args:
            src_arg_typ = next((typ for typ, name in src.args if name == arg_name), None)
            idx = next((idx for idx, (_, name) in enumerate(hyp_args) if name == arg_name), None)
            if stats is not None:
                stats.arg_name_kind.add([(idx is not None,
                                          "in_src" if src_arg_typ is not None else "not_in_src")])
            missing[arg_name] = (idx is None)
            if idx is not None:
                hyp_arg_typ, _ = hyp_args[idx]
                correct_arg_types[arg_name] = self._compare_type(tgt_arg_type, hyp_arg_typ)
                correct_arg_types_strict[arg_name] = self._compare_type(tgt_arg_type, hyp_arg_typ, cv_qualifiers=True)
                if src_arg_typ is not None:
                    pointer_check_types[arg_name] = (src_arg_typ, tgt_arg_type, hyp_arg_typ)
                del hyp_args[idx]
            if stats is not None:
                stats.arg_type_kind.add([(correct_arg_types.get(arg_name, False),
                                          "in_src" if src_arg_typ is not None else "not_in_src")])
        correct_pointer_conversion: Dict[str, Tuple[bool, bool]] = {}
        for arg_name, (src_typ, tgt_typ, hyp_typ) in pointer_check_types.items():
            if not src_typ.pointer_layer:
                correct_pointer_conversion[arg_name] = (tgt_typ.pointer_layer > 0, hyp_typ.pointer_layer > 0)
        return self.EvalOutput(func_name=correct_func_name,
                               ret_type=correct_ret_type,
                               ret_type_strict=correct_ret_type_strict,
                               missing_args=missing,
                               redundant_args=[name for _, name in hyp_args],
                               arg_types=correct_arg_types,
                               arg_types_strict=correct_arg_types_strict,
                               pointer_conversion=correct_pointer_conversion)

    class EvalScore(TypedDict):
        func_name: int  # 1 for correct, -1 for incorrect
        ret_type: int  # 1 for correct, -1 for incorrect
        ret_type_strict: int  # 1 for correct
        arg_name: int  # +1 for each correct, -1 for each missing or redundant
        arg_type: int  # +1 for each correct, -1 for each missing or incorrect
        arg_types_strict: int  # same as above
        pointer_conversion: int  # +1 for each true positive, -1 for each missing or false positive

    def _get_score(self, eval_output: 'Evaluator.EvalOutput') -> EvalScore:
        def _(x: bool) -> int:
            return 1 if x else -1

        tgt_args = list(eval_output.missing_args.keys())
        arg_name_score = sum(_(not eval_output.missing_args[x]) for x in tgt_args) - len(eval_output.redundant_args)
        arg_type_score = sum(_(eval_output.arg_types.get(x, False)) for x in tgt_args)
        arg_type_strict_score = sum(_(eval_output.arg_types_strict.get(x, False)) for x in tgt_args)
        pointer_score = 0
        for x in tgt_args:
            if x not in eval_output.pointer_conversion:
                pointer_score -= 1
            else:
                g, p = eval_output.pointer_conversion[x]
                if p:
                    pointer_score += _(g)
        return self.EvalScore({
            "func_name": _(eval_output.func_name),
            "ret_type": _(eval_output.ret_type),
            "ret_type_strict": _(eval_output.ret_type_strict),
            "arg_name": arg_name_score,
            "arg_type": arg_type_score,
            "arg_type_strict": arg_type_strict_score,
            "pointer_conversion": pointer_score,
        })

    C_KEYWORDS = {
        "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern",
        "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return", "short",
        "signed", "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while",
    }

    @classmethod
    def _prettify(cls, tokens: List[str]) -> str:
        lines = []
        indent = 0
        line = []

        def add_space(left: str, right: str) -> bool:
            if left in ["(", "!", "~", "[", ".", "->"]: return False
            if right in [")", ";", ",", "[", "]", ".", "->"]: return False
            if left == "*" == right: return False
            if right == "(" and left.isidentifier() and left not in cls.C_KEYWORDS: return False
            return True

        def newline():
            nonlocal line
            line_with_spaces = []
            for left, right in zip(line, line[1:]):
                line_with_spaces.append(left)
                if add_space(left, right):
                    line_with_spaces.append(" ")
            line_with_spaces.append(line[-1])
            lines.append("  " * indent + "".join(line_with_spaces))
            line = []

        paren_layer = 0
        for idx, (token, lookahead) in enumerate(itertools.zip_longest(tokens, tokens[1:])):
            line.append(token)
            if token == "(":
                paren_layer += 1
            elif token == ")":
                paren_layer -= 1
            elif token == "{":
                newline()
                indent += 1
            elif token == "}":
                indent -= 1
                if lookahead not in [";", "else"]:
                    newline()
            elif token == ";":
                if paren_layer == 0 and lookahead not in [";"]:
                    newline()
        return "\n".join(lines)

    @classmethod
    def _generate_code_section(cls, name: str, code: List[str], sig: FuncSignature,
                               additional: Optional[List[str]] = None) -> List[str]:
        parse_result = [
            "Function name: " + Markdown.code(sig.name),
            "Return type: " + Markdown.code(" ".join(sig.ret_type.type)),
        ]
        if len(sig.args) > 0:
            parse_result.append("Arguments: \n" + Markdown.list([
                Markdown.code(name) + ": " + Markdown.code(" ".join(typ.type)) for typ, name in sig.args], indent=2))
        if additional is not None:
            parse_result += additional
        return [
            Markdown.bold(name),
            Markdown.code_block(cls._prettify(code)),
            Markdown.list(parse_result),
        ]

    KEY_DESCRIPTION = {
        "unparsable": "Unparsable function signature",
        "arg_name": "Argument name",
        "arg_type": "Argument type (ignoring CV)",
        "arg_type_strict": "Argument type",
        "func_name": "Function name",
        "pointer_conversion": "Pointer conversion",
        "ret_type": "Return type (ignoring CV)",
        "ret_type_strict": "Return type",
    }

    @classmethod
    def _generate_code_and_metrics(cls, name: str, code: List[str], sig: FuncSignature,
                                   is_correct: 'Evaluator.EvalOutput',
                                   additional: Optional[List[str]] = None) -> List[str]:
        arg_list: List[str] = []
        redundant = is_correct.redundant_args.copy()
        for arg_typ, arg_name in reversed(sig.args):
            # Go in reverse order so we mark later occurrences of the same variable as redundant.
            name_str = Markdown.code(arg_name)
            type_str = Markdown.code(" ".join(arg_typ.type))
            if arg_name in redundant:
                name_str = name_str + " " + Markdown.bool(False, "(redundant)")
                redundant.remove(arg_name)
            else:
                type_str = Markdown.bool(is_correct.arg_types[arg_name], type_str)
            arg_list.append(name_str + ": " + type_str)
        arg_list = list(reversed(arg_list))
        missing = [k for k, v in is_correct.missing_args.items() if v]
        if len(missing) > 0:
            arg_list.append(Markdown.underline("Missing:") + " " +
                            Markdown.bool(False, ", ".join(Markdown.code(v) for v in missing)))

        parse_result = [
            "Function name: " + Markdown.bool(is_correct.func_name, Markdown.code(sig.name)),
            "Return type: " + Markdown.bool(is_correct.ret_type, Markdown.code(" ".join(sig.ret_type.type))),
        ]
        if len(arg_list) > 0:
            parse_result.append("Arguments:\n" + Markdown.list(arg_list, indent=2))
        if additional is not None:
            parse_result += additional
        return [
            Markdown.bold(name),
            Markdown.code_block(cls._prettify(code)),
            Markdown.list(parse_result),
        ]

    class HypOutput(NamedTuple):
        tokens: List[str]
        func_sig: FuncSignature
        is_correct: 'EvalOutput'
        scores_diff: Dict[str, int]

    def _generate_markdown_section(
            self, idx: int,
            src_tokens: List[str], src_func_sig: FuncSignature,
            tgt_tokens: List[str], tgt_func_sig: FuncSignature,
            hyp_output: Dict[str, HypOutput],
            overlap_scores: Dict[str, float],
            repo: Optional[str] = None, sha: Optional[str] = None) -> str:

        outputs: List[List[str]] = [
            self._generate_code_section("Decompiled (source)", src_tokens, src_func_sig),
            self._generate_code_section("Original (target)", tgt_tokens, tgt_func_sig),
        ]
        if repo is not None or sha is not None:
            outputs.insert(0, [
                Markdown.bold("Metadata"),
                Markdown.list([
                    *(["Repository: " + Markdown.link(repo, f"https://github.com/{repo}")] if repo is not None else []),
                    *([f"Binary Hash: {sha}"] if sha is not None else []),
                ])
            ])

        for name, hyp in hyp_output.items():
            bleu4 = tx.evals.sentence_bleu([tgt_tokens], hyp.tokens, max_order=4, smooth=True)
            bleu8 = tx.evals.sentence_bleu([tgt_tokens], hyp.tokens, max_order=8, smooth=True)
            improvements = [key for key, diff in hyp.scores_diff.items() if diff > 0]
            deteriorates = [key for key, diff in hyp.scores_diff.items() if diff < 0]
            score = overlap_scores[name]
            additional_evals = [
                f"BLEU4 = {bleu4:.2f}, BLEU8 = {bleu8:.2f}",
                f"Similarity Score: " + (f'<div class="highlight">{score:.3f}</div>'
                                         if score >= 0.8 else f"{score:.3f}"),
            ]
            for list_name, items in [("Improvements", improvements), ("Deteriorations", deteriorates)]:
                tag = Markdown.underline(f"{list_name} w.r.t Decompiled Code:")
                items = [Markdown.link(self.KEY_DESCRIPTION[key], "#" + self._get_list_id(key, name)) for key in items]
                list_str = "; ".join(items) if len(items) > 0 else "(None)"
                additional_evals.append(tag + " " + list_str)
            outputs.append(self._generate_code_and_metrics(
                f"Prediction ({name})", hyp.tokens, hyp.func_sig, hyp.is_correct, additional_evals))

        section = Markdown.collapse_section(
            f"Example {idx}:", f"example-{idx}",
            list(itertools.chain.from_iterable(outputs)))
        return "\n\n".join(section)

    def _parse_raw_code(self, code: str, syntax_correct: bool = True) -> Tuple[List[str], FuncSignature, bool]:
        tokens = self._split_code(code, syntax_correct=syntax_correct)
        parsable = True
        try:
            func_sig = self._parse_func(tokens, syntax_correct=syntax_correct)
        except:
            parsable = False
            func_sig = FuncSignature(TypeSignature(["<parse failed>"], False), "<parse failed>", [])
        return tokens, func_sig, parsable

    def add(self, src: str, tgt: str, hyps: Dict[str, str], overlap_scores: Dict[str, float],
            repo: Optional[str] = None, sha: Optional[str] = None) -> None:
        src_tokens, src_func_sig, src_parsable = self._parse_raw_code(src)
        tgt_tokens, tgt_func_sig, _ = self._parse_raw_code(tgt)
        self.references.append(tgt_tokens)

        is_correct_src = self._evaluate_signatures(src_func_sig, tgt_func_sig, src_func_sig)
        scores_src = self._get_score(is_correct_src)

        hyp_outputs = OrderedDict()
        for name, stats in self.stats.items():
            if name != self.DECOMPILED_NAME:
                hyp = hyps[name]
                hyp_tokens, hyp_func_sig, parsable = self._parse_raw_code(hyp, syntax_correct=False)

                if not parsable:
                    stats.deteriorated_examples["unparsable"].append(self.index)
                is_correct_hyp = self._evaluate_signatures(src_func_sig, tgt_func_sig, hyp_func_sig, stats)
                scores_hyp = self._get_score(is_correct_hyp)
                scores_diff = {key: sign(scores_hyp[key] - scores_src[key]) for key in scores_hyp.keys()}
                for key, diff in scores_diff.items():
                    stats.improving.add([(key, diff)])
                    if diff == -1:
                        stats.deteriorated_examples[key].append(self.index)

                hyp_outputs[name] = self.HypOutput(hyp_tokens, hyp_func_sig, is_correct_hyp, scores_diff)
            else:
                hyp_tokens, hyp_func_sig, parsable = src_tokens, src_func_sig, src_parsable
                is_correct_hyp = is_correct_src

            self.hypotheses[name].append(hyp_tokens)
            stats.unparsable.add([not parsable])
            stats.fn_name.add([is_correct_hyp.func_name])
            stats.fn_ret_type.add([is_correct_hyp.ret_type])
            stats.fn_ret_type_strict.add([is_correct_hyp.ret_type_strict])
            stats.arg_missing.add(is_correct_hyp.missing_args.values())
            stats.arg_name.add(not v for v in is_correct_hyp.missing_args.values())
            stats.arg_type.add(is_correct_hyp.arg_types.values())
            stats.arg_type_strict.add(is_correct_hyp.arg_types_strict.values())
            stats.redundant_args += len(is_correct_hyp.redundant_args)
            for g, p in is_correct_hyp.pointer_conversion.values():
                stats.pointer.add(gold=[g], pred=[p])

        if self.export_path is not None:
            section = self._generate_markdown_section(
                self.index, src_tokens, src_func_sig, tgt_tokens, tgt_func_sig, hyp_outputs, overlap_scores,
                repo, sha)
            self.export_sections.append(section)
            self.index += 1

    def _get_list_id(self, key: str, name: str) -> str:
        # key: key of deteriorating list; name: name of dataset
        return f"list-{key}-{self.name_ids[name]}"

    def print_summary(self) -> None:
        summary_table_col_headers = [
            "Metric",
            colored("BLEU4", "green"),
            colored("BLEU8", "green"),
            colored("Unparsable function signature", "red"),
            colored("Correct func names", "green"),
            colored("Correct return types (ignoring CV)", "green"),
            colored("Correct return types (strict)", "green"),
            colored("Correct argument names", "green"),
            colored("Correct argument types (ignoring CV)", "green"),
            colored("Correct argument types (strict)", "green"),
            colored("Missing arguments", "red"),
            colored("Redundant arguments", "red"),
            colored("Pointer conversion", "green"),
        ]
        summary_table_cols: List[List[str]] = [[
            name,
            f"{tx.evals.corpus_bleu([[tgt] for tgt in self.references], self.hypotheses[name], max_order=4):.2f}",
            f"{tx.evals.corpus_bleu([[tgt] for tgt in self.references], self.hypotheses[name], max_order=8):.2f}",
            str(stats.unparsable),
            str(stats.fn_name),
            str(stats.fn_ret_type),
            str(stats.fn_ret_type_strict),
            str(stats.arg_name),
            str(stats.arg_type),
            str(stats.arg_type_strict),
            str(stats.arg_missing),
            str(stats.redundant_args),
            f"P: {stats.pointer.precision}, R: {stats.pointer.recall}",
        ] for name, stats in self.stats.items()]
        summary_table: List[List[str]] = list(zip(*([summary_table_col_headers] + summary_table_cols)))  # transpose
        summary_table_str = Markdown.table(summary_table, ["left"] + ["right"] * len(summary_table_cols))

        improvement_tables: Dict[str, str] = OrderedDict()
        for name in self.names:
            improving_table = [["Metric", "Deteriorated (↓)", "Same (-)", "Improved (↑)"]]
            for key, group in self.stats[name].improving.group_by(lambda xs: xs[0]).items():
                values = {diff: count for (_, diff), count in group}
                total = sum(values.values())
                improving_table.append([self.KEY_DESCRIPTION[key]] +
                                       [f"{values.get(diff, 0)} / {total}" for diff in [-1, 0, 1]])
            improving_table_str = Markdown.table(improving_table, ["left", "right", "right", "right"])
            improvement_tables[name] = improving_table_str

        print(summary_table_str, end='\n\n')
        for name in self.names:
            stats = self.stats[name]
            print(colored(f"{name}:", "blue"))
            print(colored("  Improving:", "yellow"), improvement_tables[name], sep='\n', end='\n\n')
            print(colored("  Arg name categories:", "yellow"),
                  stats.arg_name_kind.to_string(lambda xs: xs[1]), sep='\n')
            print(colored("  Arg type categories:", "yellow"),
                  stats.arg_type_kind.to_string(lambda xs: xs[1]), sep='\n')

        if self.export_path is not None:
            style = r"""
                div.correct {
                  display: inline-block;
                  color: green;
                }
                div.correct::after {
                  content: " ✓";
                }
                div.wrong {
                  display: inline-block;
                  color: red;
                }
                div.wrong::after {
                  content: " ✗";
                }
                div.sourceCode > pre > code {
                  white-space: pre-wrap;
                }
                div.collapse-trigger {
                  cursor: pointer;
                }
                div.collapse-trigger[collapsed='']::after {
                  content: "   ▶";
                }
                div.collapse-trigger::after {
                  content: "   ▼";
                }
                div.collapse[collapsed=''] {
                  display: none;
                }
                div.highlight {
                  display: inline-block;
                  color: red;
                }
                table, th, td {
                  border: 1px solid black;
                }
            """
            script = r"""
            function toggle() {
                let collapseId = this.getAttribute("collapse");
                let elem = document.querySelector("div.collapse[collapse='" + collapseId + "']");
                if (this.hasAttribute("collapsed")) {
                    elem.removeAttribute("collapsed");
                    this.removeAttribute("collapsed");
                } else {
                    elem.setAttribute("collapsed", "");
                    this.setAttribute("collapsed", "");
                }
        
            }
            let elems = document.querySelectorAll("div.collapse-trigger");
            for (let i = 0; i < elems.length; ++i)
                elems[i].onclick = toggle;
            """

            sections = []
            # Custom style definitions
            sections += ["<style>\n" + style + "\n</style>"]
            #
            sections += [
                "## Summary",
                "### Metric Values",
                Markdown.strip_colored(summary_table_str),
                "### Improvement w.r.t Decompiled Code",
            ]
            for name, table in improvement_tables.items():
                sections += [f"#### {name}", table]
            # Go-to button
            sections += [
                r"""**Go to:** 
                <input id="goto-id" placeholder="Enter Example ID...">
                <button onclick="window.location.hash='example-'+document.getElementById('goto-id').value">
                  Go!
                </button>"""
            ]
            # List of IDs for deteriorated examples
            sections += ["## Lists of Deteriorated Examples"]
            for name in self.names:
                stats = self.stats[name]
                example_list = [
                    "\n\n".join(Markdown.collapse_section(
                        self.KEY_DESCRIPTION[key], self._get_list_id(key, name),
                        [", ".join(Markdown.link(ex_id, f"#example-{ex_id}") for ex_id in example_ids)
                         if len(example_ids) > 0 else "(None)"])
                    ) for key, example_ids in stats.deteriorated_examples.items()]
                sections += [f"### {name}"] + example_list
            # All examples
            sections += [
                "## Examples",
                *self.export_sections,
            ]
            # JavaScript for collapsing sections
            sections += [
                '<script type="text/javascript">\n' + script + "\n</script>",
            ]
            with open(self.export_path + ".md", "w") as f:
                f.write("\n\n".join(sections))
            result = flutes.run_command([
                "pandoc", "+RTS", "-K100000000", "-RTS",  # increase stack size
                "--from", "gfm", "--to", "html", "--standalone",
                "--metadata", "title:Code Translation Evaluation",
                self.export_path + ".md", "--output", self.export_path], return_output=True)
            if len(result.captured_output) > 0:
                print(colored(result.captured_output, "red"))
            print(colored(f"Generated output at {self.export_path}", "green"))


class InputData(NamedTuple):
    names: List[str]
    src_data: List[str]
    tgt_data: List[str]
    hyp_data: Dict[str, List[str]]
    overlap_scores: Dict[str, List[float]]
    additional_data: List[Tuple[str, str, str]]  # (score, repo, sha)


def main():
    args = Args()
    flutes.register_ipython_excepthook()
    with open(args.test_file, "rb") as f:
        data = InputData(*pickle.load(f))

    for file_name, max_size in [("eval-small.html", 100), ("eval.html", len(data.src_data))]:
        evaluator = Evaluator(data.names, export=os.path.join(args.output_dir, file_name))
        for idx in trange(max_size):
            evaluator.add(src=data.src_data[idx],
                          tgt=data.tgt_data[idx],
                          hyps={name: data.hyp_data[name][idx] for name in data.names},
                          overlap_scores={name: data.overlap_scores[name][idx] for name in data.names},
                          repo=data.additional_data[idx][1], sha=data.additional_data[idx][2])
        evaluator.print_summary()


if __name__ == '__main__':
    main()
