import copy
import itertools
import json
import os
import pickle
import string
from collections import Counter, OrderedDict, defaultdict
from typing import Any, Callable, Dict, Generic, Iterable, List, Mapping, NamedTuple, Optional, Set, Tuple, TypeVar, \
    overload

import flutes
import numpy as np
import texar.torch as tx
from argtyped import Arguments
from termcolor import colored
from tqdm import trange
from typing_extensions import Literal

from cotra.parse import LexToken, Lexer


class Args(Arguments):
    test_file: str = "test_output.pkl"
    output_dir: str = "eval-webapp/application/static/data/"


T = TypeVar('T')
R = TypeVar('R')
Tokens = List[str]
JSON = Dict[str, Any]


class Portion:
    @overload
    def __init__(self, correct: int = 0, total: int = 0):
        ...

    @overload
    def __init__(self, is_correct: Iterable[bool]):
        ...

    def __init__(self, *args, **kwargs):
        def _get(pos: int, name: str, *_default: T) -> T:
            if len(args) > pos:
                return args[pos]
            if name in kwargs:
                return kwargs[name]
            if len(_default) == 1:
                return _default[0]
            raise ValueError

        if len(args) + len(kwargs) == 1 and hasattr(_get(0, "is_correct"), "__iter__"):
            iterable = _get(0, "is_correct")
            self.correct = self.total = 0
            for elem in iterable:
                self.correct += int(elem)
                self.total += 1
        else:
            if len(args) + len(kwargs) > 2:
                raise ValueError
            self.correct = _get(0, "correct", 0)
            self.total = _get(1, "total", 0)

    def __add__(self, other: 'Portion') -> 'Portion':
        return Portion(self.correct + other.correct, self.total + other.total)

    def __float__(self) -> float:
        return self.correct / self.total

    def __repr__(self) -> str:
        return f"{self.correct} / {self.total}"

    def to_json(self) -> JSON:
        return {
            "correct": self.correct,
            "total": self.total,
        }


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
            groups: Dict[Optional[R], List[Tuple[T, int]]] = self.group_by(group_fn)
        else:
            groups = {None: list(self.counter.items())}
        strings = []
        for key, group in sorted(groups.items()):
            total = sum(v for _, v in group)
            vals = ", ".join(f"{k}: {v} / {total}" for k, v in sorted(group))
            strings.append((str(key) + ": " if key is not None else "") + vals)
        return "\n".join(strings)


class ConfusionMat:
    def __init__(self, *, gold: Optional[Iterable[bool]] = None, pred: Optional[Iterable[bool]] = None):
        self.matrix = np.zeros((2, 2), dtype=np.int)
        if gold is None:
            gold = itertools.repeat(True)
        for g, p in zip(gold, pred or []):
            self.matrix[int(g), int(p)] += 1

    def __add__(self, other: 'ConfusionMat') -> 'ConfusionMat':
        result = copy.deepcopy(self)
        result.matrix += other.matrix
        return result

    def __str__(self):
        return f"P: {self.precision}, R: {self.recall}"

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
    def precision(self) -> Portion:
        return Portion(self.true_positive, self.true_positive + self.false_positive)

    @property
    def recall(self) -> Portion:
        return Portion(self.true_positive, self.true_positive + self.false_negative)

    @property
    def f1(self) -> float:
        precision = float(self.precision)
        recall = float(self.recall)
        return 2 * precision * recall / (precision + recall)

    def to_json(self) -> JSON:
        return {
            "true_positive": int(self.true_positive),
            "true_negative": int(self.true_negative),
            "false_positive": int(self.false_positive),
            "false_negative": int(self.false_negative),
        }


class Markdown:
    @classmethod
    def to_id(cls, s: str) -> str:
        valid_chars = string.ascii_lowercase + string.digits + "_-"
        return "".join(filter(lambda x: x in valid_chars, s.lower().replace("_", "-").replace(" ", "-")))

    class Table:
        def __init__(self, table: List[List[str]], align: Optional[List[str]] = None):
            if len(table) == 0 or len(table[0]) == 0:
                raise ValueError("Table must not be empty")
            if any(len(table[0]) != len(row) for row in table[1:]):
                raise ValueError("All rows must have the same number of columns")

            self.table = [[str(value) for value in row] for row in table]
            self.n_rows = len(self.table)
            self.n_cols = len(self.table[0])
            self.colors: Dict[Tuple[int, int], str] = {}

            if align is None:
                align = ["left"] * self.n_cols
            if len(align) != self.n_cols:
                raise ValueError("`align` must either be `None`, or a list with length equal to the number of columns")
            if any(align_col not in ["left", "right", "center"] for align_col in align):
                raise ValueError("Invalid align mode. Only 'left', 'right', 'center' are supported")
            self.align = align

        def set_color(self, row: int, col: int, color: str) -> None:
            self.colors[(row, col)] = color

        def to_str(self, show_colors: bool = False) -> str:
            rows = []
            width = [max(len(row[idx]) for row in self.table) for idx in range(self.n_cols)]
            for row_idx, table_row in enumerate(self.table):
                row = []
                for col_idx, value in enumerate(table_row):
                    if self.align[col_idx] == "right":
                        value = value.rjust(width[col_idx])
                    elif self.align[col_idx] == "center":
                        value = value.center(width[col_idx])
                    else:
                        value = value.ljust(width[col_idx])
                    if show_colors and (row_idx, col_idx) in self.colors:
                        value = colored(value, self.colors[row_idx, col_idx])
                    row.append(value)
                rows.append(" | ".join(row))
            rules = []
            for col_idx in range(len(width)):
                if self.align[col_idx] == "right":
                    rule = "-" * (width[col_idx] - 1) + ":"
                elif self.align[col_idx] == "center":
                    rule = ":" + "-" * (width[col_idx] - 2) + ":"
                else:
                    rule = "-" * width[col_idx]
                rules.append(rule)
            lines = [rows[0], " | ".join(rules)] + rows[1:]
            return "\n".join("| " + line + " |" for line in lines)

    @classmethod
    def indent(cls, text: str, indent: int) -> str:
        indent_str = " " * indent
        return "\n".join(indent_str + line for line in text.split("\n"))


class Stats:
    MetricTypes = Literal['int', 'float', 'portion', 'confusion_mat']

    class Metric(NamedTuple):
        key: str
        name: str
        type: 'Stats.MetricTypes'
        formatter: Optional[Dict[str, Any]] = None
        higher_is_better: Optional[bool] = None
        display_in_summary: bool = True
        display_in_example: bool = False

    METRICS = [
        Metric("bleu4", "BLEU4", type="float", formatter={"fixed": 2}, higher_is_better=True, display_in_example=True),
        Metric("bleu8", "BLEU8", type="float", formatter={"fixed": 2}, higher_is_better=True, display_in_example=True),
        Metric("bleu4_no_var", "BLEU4 (ignoring identifiers)", type="float",
               formatter={"fixed": 2}, higher_is_better=True, display_in_example=True),
        Metric("overlap_score", "Similarity score", type="float",
               formatter={"fixed": 3}, display_in_summary=False, display_in_example=True),
        Metric("unparsable", "Unparsable function signature", type="portion", higher_is_better=False),
        Metric("func_name", "Correct function names", type="portion", higher_is_better=True),
        Metric("ret_type", "Correct return types (ignoring CV)", type="portion", higher_is_better=True),
        Metric("ret_type_strict", "Correct return types (strict)", type="portion", higher_is_better=True),
        # correct argument names (w.r.t arguments in target)
        Metric("arg_name", "Correct argument names", type="portion", higher_is_better=True),
        # correct argument types (ignoring cv-qualifiers)
        Metric("arg_type", "Correct argument types (ignoring CV)", type="portion", higher_is_better=True),
        Metric("arg_type_strict", "Correct argument types (strict)", type="portion", higher_is_better=True),
        Metric("arg_missing", "Missing arguments", type="portion", higher_is_better=False),
        # extra/duplicate arguments
        Metric("arg_redundant", "Redundant arguments", type="int", higher_is_better=False),
        Metric("str_missing", "Missing string literals", type="portion", higher_is_better=False),
        Metric("str_redundant", "Redundant string literals", type="int", higher_is_better=False),
        # correct type changes from non-pointer to pointer
        Metric("pointer_conversion", "Pointer conversion", type="confusion_mat")
    ]
    TYPE_MAP: Dict['Stats.MetricTypes', type] = {
        "int": int,
        "float": float,
        "portion": Portion,
        "confusion_mat": ConfusionMat,
    }

    bleu4: float
    bleu8: float
    bleu4_no_var: float
    overlap_score: float

    unparsable: Portion
    func_name: Portion
    ret_type: Portion
    ret_type_strict: Portion
    arg_name: Portion
    arg_type: Portion
    arg_type_strict: Portion
    arg_missing: Portion
    arg_redundant: int
    str_missing: Portion
    str_redundant: int
    pointer_conversion: ConfusionMat

    @classmethod
    def format(cls, metric: Metric, value) -> str:
        if metric.formatter is not None:
            if metric.type == "float":
                return str(round(value, metric.formatter["fixed"]))
        return str(value)

    def __init__(self, initialize: bool = True):
        if initialize:
            for metric in self.METRICS:
                setattr(self, metric.key, self.TYPE_MAP[metric.type]())

    def add(self, other: 'Stats') -> None:
        for metric in self.METRICS:
            setattr(self, metric.key, getattr(self, metric.key) + getattr(other, metric.key))


class CProcessor:
    class LineParser:
        BRACES_MAP = {
            "{": ("{}", 1),
            "}": ("{}", -1),
            "(": ("()", 1),
            ")": ("()", -1),
            "[": ("[]", 1),
            "]": ("[]", -1),
        }

        def __init__(self, code: Tokens):
            self.code = code

        def find_next(self, token: str, start: int = 0) -> int:
            # Find first occurrence of `token` in `[start, len(code))`.
            return next(idx for idx in range(start, len(self.code)) if self.code[idx] == token)

        def find_prev(self, token: str, end: int) -> int:
            # Find last occurrence of `token` in `[0, end]`
            if end < 0:
                end += len(self.code)
            return next(idx for idx in range(end, -1, -1) if self.code[idx] == token)

        def check_balance(self, indices: Iterable[int], callback: Callable[[int, bool, Dict[str, int]], T]) \
                -> Optional[T]:
            # callback: (idx, is_balanced, balance) -> ret?
            balance: Dict[str, int] = defaultdict(int)
            for idx in indices:
                if self.code[idx] in self.BRACES_MAP:
                    kind, delta = self.BRACES_MAP[self.code[idx]]
                    balance[kind] += delta
                is_balance = all(v == 0 for v in balance.values())
                ret = callback(idx, is_balance, balance)
                if ret is not None:
                    return ret
            return None

        def _find_match(self, indices: Iterable[int], token: str) -> int:
            ret = self.check_balance(indices, lambda idx, is_bal, _: idx if is_bal else None)
            assert ret is not None and self.code[ret] == token
            return ret

        def find_match_left(self, braces: str, r_pos: int) -> int:
            # Find matching left brace given right position.
            if r_pos < 0:
                r_pos += len(self.code)
            assert len(braces) == 2 and self.code[r_pos] == braces[1]
            return self._find_match(range(r_pos, -1, -1), braces[0])

        def find_match_right(self, braces: str, l_pos: int) -> int:
            # Find matching right brace given left position.
            assert len(braces) == 2 and self.code[l_pos] == braces[0]
            return self._find_match(range(l_pos, len(self.code)), braces[1])

        def find_token_within(self, token: str, l: int, r: int) -> List[int]:
            # Find occurrences of `token` within `[l, r]` that are not enclosed by braces.
            indices = []
            self.check_balance(
                range(l, r + 1), lambda idx, is_bal, _:
                indices.append(idx) if self.code[idx] == token and is_bal else None)  # type: ignore[func-returns-value]
            return indices

    class TypeSignature(NamedTuple):
        type: List[str]
        pointer_layer: int  # `char **p` => 2 layers

    class FuncSignature(NamedTuple):
        ret_type: 'CProcessor.TypeSignature'
        name: str
        args: List[Tuple['CProcessor.TypeSignature', str]]  # [(type, name)]

    @classmethod
    def parse_func(cls, code: Tokens, syntax_correct: bool = True) -> FuncSignature:
        line = cls.LineParser(code)

        def parse_vardef(l: int, r: int) -> Tuple['CProcessor.TypeSignature', str]:
            # Parse a variable definition within `[l, r]` and return the type and variable name.
            # The variable name is (usually?) the leftmost identifier within the outermost '(' group.
            # If no '('s exist, then the variable name is the rightmost identifier (`type var;`).
            # The type is the rest after deleting the identifier.
            if code[l:(r + 1)] == ["..."]:
                # varargs
                return cls.TypeSignature(["..."], False), "..."

            def callback(idx: int, _, balance: Dict[str, int]) -> Optional[int]:
                if balance['[]'] == 0 and balance['{}'] == 0 and code[idx].isidentifier():
                    return idx
                return None

            lparen_pos = line.check_balance(
                range(l, r + 1), lambda idx, _, balance:
                idx if balance['[]'] == 0 and balance['{}'] == 0 and code[idx] == '(' else None)
            if lparen_pos is None:
                # Find rightmost identifier.
                index = line.check_balance(range(r, l - 1, -1), callback)
            else:
                rparen_pos = line.find_match_right("()", lparen_pos)
                index = line.check_balance(range(lparen_pos, rparen_pos + 1), callback)
            assert index is not None
            name = code[index]

            ptr_level = code[l:index].count("*")
            new_type = []
            line.check_balance(
                range(l, r + 1), lambda idx, _, balance:
                new_type.append(code[idx])  # type: ignore[func-returns-value]
                if idx != index and (code[idx] in "[]" or balance['[]'] == 0) else None)
            return cls.TypeSignature(new_type, ptr_level), name

        if syntax_correct:
            assert code[-1] == "}"
            # Find matching '{' for the final '}'
            # These enclose the function body and everything before is the signature.
            body_lbrace_pos = line.find_match_left("{}", -1)
        else:
            # The generated code might not be syntactically correct.
            body_lbrace_pos = line.find_next("{")
        # Find the first ')' before.
        arg_rparen_pos = line.find_prev(")", body_lbrace_pos)
        arg_lparen_pos = line.find_match_left("()", arg_rparen_pos)

        # Function name and return type is represented as a variable declaration before '('.
        ret_type, func_name = parse_vardef(0, arg_lparen_pos - 1)
        # Arguments are separated by ',' within arguments '()'.
        if arg_lparen_pos + 1 == arg_rparen_pos or code[arg_lparen_pos + 1] == "void":
            args = []
        else:
            if arg_rparen_pos == body_lbrace_pos - 1:
                comma_pos = line.find_token_within(",", arg_lparen_pos + 1, arg_rparen_pos - 1)
                arg_boundaries = [arg_lparen_pos] + comma_pos + [arg_rparen_pos]
            else:
                # This might be the legacy K&R-style definition.
                semicolon_pos = line.find_token_within(";", arg_rparen_pos + 1, body_lbrace_pos - 1)
                arg_boundaries = [arg_rparen_pos] + semicolon_pos
            args = [parse_vardef(l + 1, r - 1) for l, r in zip(arg_boundaries, arg_boundaries[1:])]

        if ret_type.type == [""]:
            ret_type = cls.TypeSignature(["void"], 0)  # default return type is void
        return cls.FuncSignature(ret_type, func_name, args)

    lexer = Lexer()

    @classmethod
    def tokenize(cls, code: str, syntax_correct: bool = True) -> List[str]:
        # Replace `<unk>` with `__unk__` so it can still be part of an identifier.
        code = code.replace("<unk>", "__unk__")
        return cls.lexer.lex(code)

    @classmethod
    def get_token_types(cls, code: str) -> List[LexToken]:
        # Replace `<unk>` with `__unk__` so it can still be part of an identifier.
        code = code.replace("<unk>", "__unk__")
        return list(cls.lexer.lex_tokens(code))

    @classmethod
    def is_same_type(cls, a: TypeSignature, b: TypeSignature,
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

    C_KEYWORDS = {
        "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern",
        "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return", "short",
        "signed", "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while",
    }

    @classmethod
    def prettify(cls, tokens: List[str]) -> str:
        lines = []
        indent = 0
        line: Tokens = []

        def add_space(left: str, right: str) -> bool:
            if left in ("(", "!", "~", "[", ".", "->"): return False
            if right in (")", ";", ",", "[", "]", ".", "->"): return False
            if left == ")" and right == "(": return False
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


class BaseExporter:
    def __init__(self, path: str, metrics: List[Stats.Metric], systems: List['Evaluator.System']):
        self.export_path = path
        self.index = 0
        self.metrics = metrics
        self.systems = systems

    def add_example(
            self,
            src_tokens: List[str], src_func_sig: CProcessor.FuncSignature,
            tgt_tokens: List[str], tgt_func_sig: CProcessor.FuncSignature,
            hyp_output: Dict[str, 'Evaluator.HypOutput'],
            var_map: Optional[Dict[str, Tuple[str, str]]] = None,
            repo: Optional[str] = None, sha: Optional[str] = None) -> None:
        raise NotImplementedError

    def generate(self, summary_stats: Mapping[str, Stats]):
        raise NotImplementedError


class JSONExporter(BaseExporter):
    def __init__(self, path: str, metrics: List[Stats.Metric], systems: List['Evaluator.System']):
        super().__init__(path, metrics, systems)
        self.examples: List[JSON] = []

    @classmethod
    def _generate_code_section(cls, code: List[str], sig: CProcessor.FuncSignature) -> JSON:
        return {
            "code": CProcessor.prettify(code),
            "func_name": sig.name,
            "ret_type": " ".join(sig.ret_type.type),
            "args": [(name, " ".join(typ.type)) for typ, name in sig.args],
        }

    @classmethod
    def _generate_arg_comparison(cls, sig: CProcessor.FuncSignature, is_correct: 'Evaluator.EvalOutput') -> JSON:
        arg_list: List[Tuple[str, str, Optional[bool]]] = []
        redundant = is_correct.redundant_args.copy()
        for arg_typ, arg_name in reversed(sig.args):
            # Go in reverse order so we mark later occurrences of the same variable as redundant.
            if arg_name in redundant:
                verdict = None  # redundant
                redundant.remove(arg_name)
            else:
                verdict = is_correct.arg_types[arg_name]
            arg_list.append((arg_name, " ".join(arg_typ.type), verdict))
        arg_list = list(reversed(arg_list))
        missing = [k for k, v in is_correct.missing_args.items() if v]
        return {
            "args": arg_list,  # override the list above in `_generate_code_section`
            "missing_args": missing,
        }

    def add_example(
            self,
            src_tokens: List[str], src_func_sig: CProcessor.FuncSignature,
            tgt_tokens: List[str], tgt_func_sig: CProcessor.FuncSignature,
            hyp_output: Dict[str, 'Evaluator.HypOutput'],
            var_map: Optional[Dict[str, Tuple[str, str]]] = None,
            repo: Optional[str] = None, sha: Optional[str] = None) -> None:
        var_map = var_map or {}
        json_dict: JSON = {
            "index": self.index,
            "meta_data": {
                "repo": repo,
                "sha": sha,
            },
            "var_map": var_map,
            "target": self._generate_code_section(tgt_tokens, tgt_func_sig),
            "predictions": {}
        }

        for key, hyp in hyp_output.items():
            tokens = hyp.tokens if key != Evaluator.DECOMPILED_KEY else src_tokens
            pred_dict = {
                **self._generate_code_section(tokens, hyp.func_sig),
                **self._generate_arg_comparison(hyp.func_sig, hyp.is_correct),
                "missing_strings": list(hyp.missing_strings),
                "redundant_strings": list(hyp.redundant_strings),
                "metrics": self._stats_to_json(hyp.metrics),
            }
            json_dict["predictions"][key] = pred_dict

        self.examples.append(json_dict)
        self.index += 1

    def _stats_to_json(self, stats: Stats) -> JSON:
        json_dict = {}
        for metric in self.metrics:
            value = getattr(stats, metric.key)
            if metric.type not in {"int", "float"}:
                value = value.to_json()
            json_dict[metric.key] = value
        return json_dict

    def generate(self, summary_stats: Mapping[str, Stats]):
        json_dict = {
            "metrics": [metric._asdict() for metric in self.metrics],
            "systems": [{**system._asdict(), "metrics": self._stats_to_json(summary_stats[system.key])}
                        for system in self.systems],
            "examples": self.examples,
        }
        with open(self.export_path, "w") as f:
            json.dump(json_dict, f)


class Evaluator:
    DECOMPILED_KEY = "decompiled"

    class System(NamedTuple):
        key: str
        name: str
        description: str
        use_var_map: bool = False

    SYSTEMS = [
        System("decompiled", "Decompiled", "Code produced by the decompiler", use_var_map=True),
        System("seq2seq_d", "Seq2seq-D", "Seq2seq + decompiler var names"),
        System("seq2seq_o", "Seq2seq-O", "Seq2seq + oracle var names"),
        System("seq2seq_d_ft", "Seq2seq-D +Finetune", "Finetuned Seq2seq + decompiler var names"),
        System("seq2seq_o_ft", "Seq2seq-O +Finetune", "Finetuned Seq2seq + oracle var names"),
        # System("tranx_d_greedy", "TranX-D-Greedy", "TranX + decompiler var names + greedy decoding"),
        # System("tranx_d_beam5", "TranX-D-Beam5", "TranX + decompiler var names + beam width 5"),
        System("tranx_o_greedy", "TranX-O-Greedy", "TranX + oracle var names + greedy decoding"),
        System("tranx_o_beam5", "TranX-O-Beam5", "TranX + oracle var names + beam width 5"),
        # System("tranx_d_greedy_ft", "TranX-D-Greedy +Finetune", "Finetuned TranX + decompiler var names + greedy decoding"),
        # System("tranx_d_beam5_ft", "TranX-D-Beam5 +Finetune", "Finetuned TranX + decompiler var names + beam width 5"),
        # System("tranx_o_greedy_ft", "TranX-O-Greedy +Finetune", "Finetuned TranX + oracle var names + greedy decoding"),
        # System("tranx_o_beam5_ft", "TranX-O-Beam5 +Finetune", "Finetuned TranX + oracle var names + beam width 5"),
    ]

    def __init__(self, exporter: Optional[BaseExporter] = None):
        self.summary_stats = OrderedDict((system.key, Stats()) for system in self.SYSTEMS)

        self.references: List[Tokens] = []
        self.references_no_var: List[Tokens] = []
        self.hypotheses: Dict[str, List[Tokens]] = {system.key: [] for system in self.SYSTEMS}
        self.hypotheses_no_var = {system.key: [] for system in self.SYSTEMS}

        self.index = 0
        self.exporter = exporter

    class EvalOutput(NamedTuple):
        func_name: bool  # is_correct
        ret_type: bool  # is_correct
        ret_type_strict: bool  # is_correct
        missing_args: Dict[str, bool]  # (name) -> is_missing
        redundant_args: List[str]  # [name]
        arg_types: Dict[str, bool]  # (name) -> is_correct
        arg_types_strict: Dict[str, bool]  # (name) -> is_correct
        pointer_conversion: Dict[str, Tuple[bool, bool]]  # (name) -> (should_convert, did_convert)

    def _evaluate_signatures(self,
                             src: CProcessor.FuncSignature,
                             tgt: CProcessor.FuncSignature,
                             hyp: CProcessor.FuncSignature) -> 'EvalOutput':
        correct_func_name = tgt.name == hyp.name
        correct_ret_type = CProcessor.is_same_type(tgt.ret_type, hyp.ret_type)
        correct_ret_type_strict = CProcessor.is_same_type(tgt.ret_type, hyp.ret_type, cv_qualifiers=True)
        missing: Dict[str, bool] = {}
        correct_arg_types: Dict[str, bool] = {}
        correct_arg_types_strict: Dict[str, bool] = {}
        pointer_check_types = {"": (src.ret_type, tgt.ret_type, hyp.ret_type)}
        hyp_args = hyp.args.copy()
        for tgt_arg_type, arg_name in tgt.args:
            src_arg_typ = next((typ for typ, name in src.args if name == arg_name), None)
            idx = next((idx for idx, (_, name) in enumerate(hyp_args) if name == arg_name), None)
            missing[arg_name] = (idx is None)
            if idx is not None:
                hyp_arg_typ, _ = hyp_args[idx]
                correct_arg_types[arg_name] = CProcessor.is_same_type(tgt_arg_type, hyp_arg_typ)
                correct_arg_types_strict[arg_name] = CProcessor.is_same_type(
                    tgt_arg_type, hyp_arg_typ, cv_qualifiers=True)
                if src_arg_typ is not None:
                    pointer_check_types[arg_name] = (src_arg_typ, tgt_arg_type, hyp_arg_typ)
                del hyp_args[idx]

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

    class HypOutput(NamedTuple):
        tokens: List[str]
        func_sig: CProcessor.FuncSignature
        is_correct: 'Evaluator.EvalOutput'
        missing_strings: Set[str]
        redundant_strings: Set[str]
        metrics: Stats

    def _parse_raw_code(self, code: str, syntax_correct: bool = True) \
            -> Tuple[List[str], CProcessor.FuncSignature, bool]:
        tokens = CProcessor.tokenize(code, syntax_correct=syntax_correct)
        parsable = True
        try:
            func_sig = CProcessor.parse_func(tokens, syntax_correct=syntax_correct)
        except:
            parsable = False
            func_sig = CProcessor.FuncSignature(
                CProcessor.TypeSignature(["<parse failed>"], False), "<parse failed>", [])
        return tokens, func_sig, parsable

    @staticmethod
    def _sign(x: int) -> int:
        return 1 if x > 0 else -1 if x < 0 else 0

    @staticmethod
    def _analyze_code(code: str) -> Tuple[List[str], Set[str]]:
        # ([token_no_var], {string_literals})
        # Tokenize code and replace all identifiers with a dummy token.
        tokens = CProcessor.get_token_types(code)
        tokens_no_var = [token.value if token.type != "ID" else "_VAR_" for token in tokens]
        string_literals = {token.value for token in tokens if token.type in {'STRING_LITERAL', 'WSTRING_LITERAL'}}
        return tokens_no_var, string_literals

    def add(self, src: str, tgt: str, hyps: Dict[str, str], overlap_scores: Dict[str, float],
            var_map: Optional[Dict[str, Tuple[str, str]]] = None,
            repo: Optional[str] = None, sha: Optional[str] = None) -> None:
        var_map = var_map or {}
        var_replaced_src = src
        for key, (_, oracle) in var_map.items():
            var_replaced_src = var_replaced_src.replace(key, oracle)
        src_tokens, src_func_sig, src_parsable = self._parse_raw_code(var_replaced_src)
        raw_src_tokens, _, _ = self._parse_raw_code(src)
        tgt_tokens, tgt_func_sig, _ = self._parse_raw_code(tgt)
        self.references.append(tgt_tokens)
        tgt_tokens_no_var, tgt_string_literals = self._analyze_code(tgt)
        self.references_no_var.append(tgt_tokens_no_var)

        hyp_outputs = OrderedDict()
        for name, summary_stats in self.summary_stats.items():
            if name != self.DECOMPILED_KEY:
                hyp = hyps[name]
                hyp_tokens, hyp_func_sig, parsable = self._parse_raw_code(hyp, syntax_correct=False)
            else:
                hyp = var_replaced_src
                hyp_tokens, hyp_func_sig, parsable = src_tokens, src_func_sig, src_parsable
            hyp_tokens_no_var, hyp_string_literals = self._analyze_code(hyp)
            is_correct_hyp = self._evaluate_signatures(src_func_sig, tgt_func_sig, hyp_func_sig)

            missing_strings = tgt_string_literals - hyp_string_literals
            redundant_strings = hyp_string_literals - tgt_string_literals

            self.hypotheses[name].append(hyp_tokens)
            self.hypotheses_no_var[name].append(hyp_tokens_no_var)
            cur_stats = Stats()
            cur_stats.bleu4 = tx.evals.sentence_bleu([tgt_tokens], hyp_tokens, max_order=4, smooth=True)
            cur_stats.bleu8 = tx.evals.sentence_bleu([tgt_tokens], hyp_tokens, max_order=8, smooth=True)
            cur_stats.bleu4_no_var = tx.evals.sentence_bleu(
                [tgt_tokens_no_var], hyp_tokens_no_var, max_order=4, smooth=True)
            cur_stats.overlap_score = overlap_scores.get(name, 0.0)
            cur_stats.unparsable = Portion([not parsable])
            cur_stats.func_name = Portion([is_correct_hyp.func_name])
            cur_stats.ret_type = Portion([is_correct_hyp.ret_type])
            cur_stats.ret_type_strict = Portion([is_correct_hyp.ret_type_strict])
            cur_stats.arg_name = Portion([not v for v in is_correct_hyp.missing_args.values()])
            cur_stats.arg_type = Portion(is_correct_hyp.arg_types.values())
            cur_stats.arg_type_strict = Portion(is_correct_hyp.arg_types_strict.values())
            cur_stats.arg_missing = Portion(is_correct_hyp.missing_args.values())
            cur_stats.arg_redundant = len(is_correct_hyp.redundant_args)
            cur_stats.str_missing = Portion(len(missing_strings), len(tgt_string_literals))
            cur_stats.str_redundant = len(redundant_strings)
            cur_stats.pointer_conversion = ConfusionMat(gold=[g for g, _ in is_correct_hyp.pointer_conversion.values()],
                                                        pred=[p for _, p in is_correct_hyp.pointer_conversion.values()])
            hyp_outputs[name] = self.HypOutput(
                hyp_tokens, hyp_func_sig, is_correct_hyp, missing_strings, redundant_strings, cur_stats)
            summary_stats.add(cur_stats)

        if self.exporter is not None:
            self.exporter.add_example(
                raw_src_tokens, src_func_sig, tgt_tokens, tgt_func_sig, hyp_outputs,
                var_map, repo, sha)

        self.index += 1

    def print_summary(self) -> None:
        summary_table_col_headers: List[Tuple[str, Optional[str]]] = [("Metric", None)]
        for metric in Stats.METRICS:
            if metric.display_in_summary:
                color = {True: "green", False: "red", None: None}[metric.higher_is_better]
                summary_table_col_headers.append((metric.name, color))
        references = [[tgt] for tgt in self.references]
        references_no_var = [[tgt] for tgt in self.references_no_var]
        summary_table_cols: List[List[str]] = []
        for name, stats in self.summary_stats.items():
            stats.bleu4 = tx.evals.corpus_bleu(references, self.hypotheses[name], max_order=4)
            stats.bleu8 = tx.evals.corpus_bleu(references, self.hypotheses[name], max_order=8)
            stats.bleu4_no_var = tx.evals.corpus_bleu(references_no_var, self.hypotheses_no_var[name], max_order=4)
            summary_table_cols.append([
                Stats.format(metric, getattr(stats, metric.key))
                for metric in Stats.METRICS if metric.display_in_summary])
        summary_table_items: List[List[str]] = list(
            map(list, zip(*([[name for name, _ in summary_table_col_headers]] + summary_table_cols))))  # transpose
        summary_table = Markdown.Table(summary_table_items, ["left"] + ["right"] * len(summary_table_cols))
        for idx, (_, color) in enumerate(summary_table_col_headers):
            if color is not None:
                summary_table.set_color(idx, 0, color)

        print(summary_table.to_str(show_colors=True), end='\n\n')

        if self.exporter is not None:
            self.exporter.generate(self.summary_stats)


class InputData(NamedTuple):
    names: List[str]
    src_data: List[str]
    tgt_data: List[str]
    hyp_data: Dict[str, List[str]]
    overlap_scores: Dict[str, List[float]]
    additional_data: List[Tuple[str, str, str, str]]  # (var_map, score, repo, sha)


def main():
    args = Args()
    # flutes.register_ipython_excepthook()
    with open(args.test_file, "rb") as f:
        data = InputData(*pickle.load(f))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    print("Please check that the configured system names matches the collected data:")
    hypothesis_systems = [system for system in Evaluator.SYSTEMS if system.key != Evaluator.DECOMPILED_KEY]
    name_map = {name: system for system, name in itertools.zip_longest(hypothesis_systems, data.names)}
    name_table = Markdown.Table(
        [["System name", "Name from data"]] +
        [[system.name, name] for name, system in name_map.items()])
    print(name_table.to_str())

    for file_name, max_size in [("eval-small", 100), ("eval", len(data.src_data))]:
        # exporter = HTMLExporter(os.path.join(args.output_dir, file_name + ".html"), data.names)
        exporter = JSONExporter(os.path.join(args.output_dir, file_name + ".json"), Stats.METRICS, Evaluator.SYSTEMS)
        evaluator = Evaluator(exporter=exporter)
        for idx in trange(max_size):
            var_names, score, repo, sha = data.additional_data[idx]
            var_map = {}
            if var_names != "":
                var_map = {name: (decomp, oracle) for name, decomp, oracle in flutes.chunk(3, var_names.split("\0"))}
            evaluator.add(src=data.src_data[idx],
                          tgt=data.tgt_data[idx],
                          hyps={name_map[name].key: data.hyp_data[name][idx] for name in data.names},
                          overlap_scores={name_map[name].key: data.overlap_scores[name][idx] for name in data.names},
                          var_map=var_map, repo=repo, sha=sha)
        evaluator.print_summary()


if __name__ == '__main__':
    main()
