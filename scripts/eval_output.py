import itertools
import json
import os
import pickle
import string
from collections import Counter, OrderedDict, defaultdict
from typing import Any, Callable, Dict, Generic, Iterable, List, Mapping, NamedTuple, Optional, Tuple, TypeVar, Set

import flutes
import numpy as np
import texar.torch as tx
from argtyped import Arguments
from termcolor import colored
from tqdm import trange
from typing_extensions import TypedDict

from cotra.parse import LexToken, Lexer


class Args(Arguments):
    test_file: str = "test_output.pkl"
    output_dir: str = "."


T = TypeVar('T')
R = TypeVar('R')
Tokens = List[str]


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


class Markdown:
    @staticmethod
    def code(s: str) -> str:
        return f"`{s}`"

    @staticmethod
    def code_block(s: str, lang: str = "c") -> str:
        return f"```{lang}\n{s.strip()}\n```"

    @classmethod
    def bold(cls, s: str) -> str:
        return f"**{s}**"

    @classmethod
    def underline(cls, s: str) -> str:
        return f"<u>{s}</u>"

    @classmethod
    def list(cls, lines: List[str], indent: int = 0, numbered: bool = False) -> str:
        indent_str = " " * indent
        if numbered:
            return "\n".join(indent_str + f"{idx}. " + line for idx, line in enumerate(lines))
        else:
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

    class Table:
        def __init__(self, table: List[List[str]], align: Optional[List[str]]):
            if len(table) == 0 or len(table[0]) == 0:
                raise ValueError("Table must not be empty")
            if any(len(table[0]) != len(row) for row in table[1:]):
                raise ValueError("All rows must have the same number of columns")

            self.table = table
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
    def link(cls, text: str, url: str) -> str:
        return f"[{text}]({url})"

    @classmethod
    def indent(cls, text: str, indent: int) -> str:
        indent_str = " " * indent
        return "\n".join(indent_str + line for line in text.split("\n"))


class Stats:
    KEYS = ["unparsable",
            "arg_name",
            "arg_type",
            "arg_type_strict",
            "func_name",
            "pointer_conversion",
            "ret_type",
            "ret_type_strict"]

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
        self.missing_strings = 0
        self.redundant_strings = 0

        self.improving = CategoryCounter()  # examples that improved compared to
        self.arg_name_kind = CategoryCounter()
        self.arg_type_kind = CategoryCounter()
        self.deteriorated_examples = {key: [] for key in self.KEYS}


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
    def __init__(self, path: str, names: List[str]):
        self.export_path = path
        self.index = 0
        self.names = names
        self.name_ids = OrderedDict((name, Markdown.to_id(name)) for name in names)

    def add_example(
            self,
            src_tokens: List[str], src_func_sig: CProcessor.FuncSignature,
            tgt_tokens: List[str], tgt_func_sig: CProcessor.FuncSignature,
            hyp_output: Dict[str, 'Evaluator.HypOutput'],
            var_map: Optional[Dict[str, Tuple[str, str]]] = None,
            repo: Optional[str] = None, sha: Optional[str] = None) -> None:
        raise NotImplementedError

    def generate(self, stats: Mapping[str, Stats], summary_table: Markdown.Table,
                 improvement_tables: Mapping[str, Markdown.Table]):
        raise NotImplementedError


class HTMLExporter(BaseExporter):
    def __init__(self, path: str, names: List[str]):
        super().__init__(path, names)
        self.export_sections: List[str] = []
        self.var_maps: List[Dict[str, Tuple[str, str]]] = []

    @classmethod
    def _generate_code_section(cls, name: str, code: List[str], sig: CProcessor.FuncSignature,
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
            Markdown.code_block(CProcessor.prettify(code)),
            Markdown.list(parse_result),
        ]

    @classmethod
    def _generate_code_and_metrics(cls, name: str, code: List[str], sig: CProcessor.FuncSignature,
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
            Markdown.code_block(CProcessor.prettify(code)),
            Markdown.list(parse_result),
        ]

    def add_example(
            self,
            src_tokens: List[str], src_func_sig: CProcessor.FuncSignature,
            tgt_tokens: List[str], tgt_func_sig: CProcessor.FuncSignature,
            hyp_output: Dict[str, 'Evaluator.HypOutput'],
            var_map: Optional[Dict[str, Tuple[str, str]]] = None,
            repo: Optional[str] = None, sha: Optional[str] = None) -> None:

        var_map = var_map or {}
        var_map = {k: [d, o] for k, (d, o) in var_map.items()}
        self.var_maps.append(var_map)
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
            improvements = [key for key, diff in hyp.scores_diff.items() if diff > 0]
            deteriorates = [key for key, diff in hyp.scores_diff.items() if diff < 0]
            bleu4 = hyp.metrics['bleu4']
            bleu8 = hyp.metrics['bleu8']
            score = hyp.metrics['overlap_score']
            additional_evals = [
                f"BLEU4 = {bleu4:.2f}, BLEU8 = {bleu8:.2f}",
                f"Similarity Score: " + (f'<div class="highlight">{score:.3f}</div>'
                                         if score >= 0.8 else f"{score:.3f}"),
            ]
            for list_name, items in [("Improvements", improvements), ("Deteriorations", deteriorates)]:
                tag = Markdown.underline(f"{list_name} w.r.t Decompiled Code:")
                items = [Markdown.link(Stats.KEY_DESCRIPTION[key], "#" + self._get_list_id(key, name)) for key in items]
                list_str = "; ".join(items) if len(items) > 0 else "(None)"
                additional_evals.append(tag + " " + list_str)
            outputs.append(self._generate_code_and_metrics(
                f"Prediction ({name})", hyp.tokens, hyp.func_sig, hyp.is_correct, additional_evals))

        section = Markdown.collapse_section(
            f"Example {self.index}:", f"example-{self.index}",
            list(itertools.chain.from_iterable(outputs)))
        section_str = "\n\n".join(section)

        self.export_sections.append(section_str)
        self.index += 1

    def _get_list_id(self, key: str, name: str) -> str:
        # key: key of deteriorating list; name: name of dataset
        return f"list-{key}-{self.name_ids[name]}"

    STYLE = r"""
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
        div.sourceCode > pre > code .decompiled-var {
          text-decoration: underline solid red;
        }
        div.sourceCode > pre > code .oracle-var {
          text-decoration: underline solid green;
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
        .hide {
          display: none !important;
        }
        button.varname-button {
          position: fixed;
          bottom: 0;
          right: 0;
          z-index: 100;
        }
    """
    SCRIPT = r"""
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
        
        function initVarNames() {
            for (let key in var_maps) {
                let value = var_maps[key];
                let elem = document.querySelector("div[collapse='example-" + key + "'] > div.sourceCode:first-of-type");
                let html = elem.innerHTML;
                for (let var_id in value) {
                    let var_names = value[var_id];
                    let decomp_span = "<span class='decompiled-var'>" + var_names[0] + "</span>";
                    let oracle_span = "<span class='oracle-var hide'>" + var_names[1] + "</span>";
                    html = html.replace(new RegExp(var_id, 'g'),  decomp_span + oracle_span);
                }
                elem.innerHTML = html;
            }
        }
        initVarNames();
        
        let currentVarName = "decompiled";
        
        function toggleVarNames() {
            document.querySelectorAll(".decompiled-var").forEach(function(item) {
                item.classList.toggle("hide");
            });
            document.querySelectorAll(".oracle-var").forEach(function(item) {
                item.classList.toggle("hide");
            });
            if (currentVarName === "decompiled") {
                currentVarName = "oracle";
                document.querySelector(".varname-button").innerText = "Switch to Decompiled var names";
            } else {
                currentVarName = "decompiled";
                document.querySelector(".varname-button").innerText = "Switch to Oracle var names";
            }
        }
    """

    def generate(self, stats: Mapping[str, Stats], summary_table: Markdown.Table,
                 improvement_tables: Mapping[str, Markdown.Table]):

        sections = []
        # Custom style definitions
        sections += ["<style>\n" + self.STYLE + "\n</style>"]
        #
        sections += [
            "## Summary",
            "### Metric Values",
            summary_table.to_str(),
            "### Improvement w.r.t Decompiled Code",
        ]
        for name, table in improvement_tables.items():
            sections += [f"#### {name}", table.to_str()]
        # Go-to button
        sections += [
            r"""**Go to:** 
            <input id="goto-id" placeholder="Enter Example ID...">
            <button onclick="window.location.hash='example-'+document.getElementById('goto-id').value">
              Go!
            </button>
            <button class="varname-button" onclick="toggleVarNames()">
              Switch to Oracle var names
            </button>
            """,
        ]
        # List of IDs for deteriorated examples
        sections += ["## Lists of Deteriorated Examples"]
        for name in self.names:
            example_list = [
                "\n\n".join(Markdown.collapse_section(
                    Stats.KEY_DESCRIPTION[key], self._get_list_id(key, name),
                    [", ".join(Markdown.link(ex_id, f"#example-{ex_id}") for ex_id in example_ids)
                     if len(example_ids) > 0 else "(None)"])
                ) for key, example_ids in stats[name].deteriorated_examples.items()]
            sections += [f"### {name}"] + example_list
        # All examples
        sections += [
            "## Examples",
            *self.export_sections,
        ]
        # JavaScript for storing variable names
        sections += [
            r'<script type="text/javascript">' + "\n".join([
                "let var_maps = {};",
                *[rf'var_maps[{idx}] = {var_map!r}' for idx, var_map in enumerate(self.var_maps)],
            ]) + r'</script>',
        ]
        # JavaScript for collapsing sections
        sections += [
            '<script type="text/javascript">\n' + self.SCRIPT + "\n</script>",
        ]
        with open(self.export_path + ".md", "w") as f:
            f.write("\n\n".join(sections))
        result = flutes.run_command([
            "pandoc", "+RTS", "-K100000000", "-RTS",  # increase stack size
            "--from", "gfm", "--to", "html", "--standalone",
            "--metadata", "title:Code Translation Evaluation",
            self.export_path + ".md", "--output", self.export_path], return_output=True)
        if result.captured_output is not None and len(result.captured_output) > 0:
            print(colored(result.captured_output.decode('utf-8'), "red"))
        print(colored(f"Generated output at {self.export_path}", "green"))


JSON = Dict[str, Any]


class JSONExporter(BaseExporter):
    def __init__(self, path: str, names: List[str]):
        super().__init__(path, names)
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
            "args": arg_list,
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
        json_dict = {
            "index": self.index,
            "meta_data": {
                "repo": repo,
                "sha": sha,
            },
            "var_map": var_map,
            "src": self._generate_code_section(src_tokens, src_func_sig),
            "tgt": self._generate_code_section(tgt_tokens, tgt_func_sig),
            "preds": []
        }

        for name, hyp in hyp_output.items():
            # improvements = [key for key, diff in hyp.scores_diff.items() if diff > 0]
            # deteriorates = [key for key, diff in hyp.scores_diff.items() if diff < 0]
            # for list_name, items in [("Improvements", improvements), ("Deteriorations", deteriorates)]:
            #     tag = Markdown.underline(f"{list_name} w.r.t Decompiled Code:")
            #     items = [Markdown.link(Stats.KEY_DESCRIPTION[key], "#" + self._get_list_id(key, name)) for key in items]
            #     list_str = "; ".join(items) if len(items) > 0 else "(None)"
            #     additional_evals.append(tag + " " + list_str)
            pred_dict = {
                "source": name,
                **self._generate_code_section(hyp.tokens, hyp.func_sig),
                **self._generate_arg_comparison(hyp.func_sig, hyp.is_correct),
                **hyp.metrics,
                "missing_strings": list(hyp.missing_strings),
                "redundant_strings": list(hyp.redundant_strings),
            }
            json_dict["preds"].append(pred_dict)

        self.examples.append(json_dict)
        self.index += 1

    def generate(self, stats: Mapping[str, Stats], summary_table: Markdown.Table,
                 improvement_tables: Mapping[str, Markdown.Table]):
        json_dict = {
            "summary": {
                "summary_table": summary_table.table,
                "improvement_tables": [(name, improvement_tables[name].table) for name in self.names],
            },
            "examples": self.examples,
        }
        with open(self.export_path, "w") as f:
            json.dump(json_dict, f)


class Evaluator:
    DECOMPILED_NAME = "Decompiled"

    def __init__(self, names: List[str], exporter: Optional[BaseExporter] = None):
        self.stats = OrderedDict((name, Stats()) for name in [self.DECOMPILED_NAME] + names)
        self.names = names
        self.name_ids = OrderedDict((name, Markdown.to_id(name)) for name in names)

        self.references: List[Tokens] = []
        self.references_no_var: List[Tokens] = []
        self.hypotheses: Dict[str, List[Tokens]] = {name: [] for name in [self.DECOMPILED_NAME] + names}
        self.hypotheses_no_var = {name: [] for name in self.hypotheses.keys()}

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
                             hyp: CProcessor.FuncSignature,
                             stats: Optional[Stats] = None):
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
            if stats is not None:
                stats.arg_name_kind.add([(idx is not None,
                                          "in_src" if src_arg_typ is not None else "not_in_src")])
            missing[arg_name] = (idx is None)
            if idx is not None:
                hyp_arg_typ, _ = hyp_args[idx]
                correct_arg_types[arg_name] = CProcessor.is_same_type(tgt_arg_type, hyp_arg_typ)
                correct_arg_types_strict[arg_name] = CProcessor.is_same_type(
                    tgt_arg_type, hyp_arg_typ, cv_qualifiers=True)
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
        arg_type_strict: int  # same as above
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
        return {
            "func_name": _(eval_output.func_name),
            "ret_type": _(eval_output.ret_type),
            "ret_type_strict": _(eval_output.ret_type_strict),
            "arg_name": arg_name_score,
            "arg_type": arg_type_score,
            "arg_type_strict": arg_type_strict_score,
            "pointer_conversion": pointer_score,
        }

    class MetricDict(TypedDict):
        bleu4: float
        bleu8: float
        overlap_score: float
        bleu4_no_var: float

    class HypOutput(NamedTuple):
        tokens: List[str]
        func_sig: CProcessor.FuncSignature
        is_correct: 'Evaluator.EvalOutput'
        scores_diff: Dict[str, int]
        missing_strings: Set[str]
        redundant_strings: Set[str]
        metrics: 'Evaluator.MetricDict'

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
        src_tokens_no_var, src_string_literals = self._analyze_code(src)
        tgt_tokens_no_var, tgt_string_literals = self._analyze_code(tgt)
        self.references_no_var.append(tgt_tokens_no_var)
        # string_literals = src_string_literals & tgt_string_literals
        string_literals = tgt_string_literals

        is_correct_src = self._evaluate_signatures(src_func_sig, tgt_func_sig, src_func_sig)
        scores_src = self._get_score(is_correct_src)

        hyp_outputs = OrderedDict()
        for name, stats in self.stats.items():
            if name != self.DECOMPILED_NAME:
                hyp = hyps[name]
                hyp_tokens, hyp_func_sig, parsable = self._parse_raw_code(hyp, syntax_correct=False)
                hyp_tokens_no_var, hyp_string_literals = self._analyze_code(hyp)
            else:
                hyp_tokens, hyp_func_sig, parsable = src_tokens, src_func_sig, src_parsable
                hyp_tokens_no_var = src_tokens_no_var
                hyp_string_literals = src_string_literals
                is_correct_hyp = is_correct_src

            missing_strings = string_literals - hyp_string_literals
            redundant_strings = hyp_string_literals - string_literals

            if name != self.DECOMPILED_NAME:
                if not parsable:
                    stats.deteriorated_examples["unparsable"].append(self.index)
                is_correct_hyp = self._evaluate_signatures(src_func_sig, tgt_func_sig, hyp_func_sig, stats)
                scores_hyp = self._get_score(is_correct_hyp)
                scores_diff = {key: self._sign(scores_hyp[key] - scores_src[key])  # type: ignore[misc]
                               for key in scores_hyp.keys()}
                for key, diff in scores_diff.items():
                    stats.improving.add([(key, diff)])
                    if diff == -1:
                        stats.deteriorated_examples[key].append(self.index)

                bleu4 = tx.evals.sentence_bleu([tgt_tokens], hyp_tokens, max_order=4, smooth=True)
                bleu8 = tx.evals.sentence_bleu([tgt_tokens], hyp_tokens, max_order=8, smooth=True)
                bleu4_no_var = tx.evals.sentence_bleu([tgt_tokens_no_var], hyp_tokens_no_var, max_order=4, smooth=True)
                metrics: 'Evaluator.MetricDict' = {
                    "bleu4": bleu4,
                    "bleu8": bleu8,
                    "overlap_score": overlap_scores[name],
                    "bleu4_no_var": bleu4_no_var,
                }
            # else:
            #     metrics = {}
                hyp_outputs[name] = self.HypOutput(
                    hyp_tokens, hyp_func_sig, is_correct_hyp, scores_diff,
                    missing_strings, redundant_strings, metrics)

            self.hypotheses[name].append(hyp_tokens)
            self.hypotheses_no_var[name].append(hyp_tokens_no_var)
            stats.unparsable.add([not parsable])
            stats.fn_name.add([is_correct_hyp.func_name])
            stats.fn_ret_type.add([is_correct_hyp.ret_type])
            stats.fn_ret_type_strict.add([is_correct_hyp.ret_type_strict])
            stats.arg_missing.add(is_correct_hyp.missing_args.values())
            stats.arg_name.add(not v for v in is_correct_hyp.missing_args.values())
            stats.arg_type.add(is_correct_hyp.arg_types.values())
            stats.arg_type_strict.add(is_correct_hyp.arg_types_strict.values())
            stats.redundant_args += len(is_correct_hyp.redundant_args)
            stats.missing_strings += len(missing_strings)
            stats.redundant_strings += len(redundant_strings)
            for g, p in is_correct_hyp.pointer_conversion.values():
                stats.pointer.add(gold=[g], pred=[p])

        if self.exporter is not None:
            self.exporter.add_example(
                raw_src_tokens, src_func_sig, tgt_tokens, tgt_func_sig, hyp_outputs,
                var_map, repo, sha)

        self.index += 1

    def print_summary(self) -> None:
        summary_table_col_headers: List[Tuple[str, Optional[str]]] = [
            ("Metric", None),
            ("BLEU4", "green"),
            ("BLEU8", "green"),
            ("BLEU4 (ignoring identifiers)", "green"),
            ("Unparsable function signature", "red"),
            ("Correct func names", "green"),
            ("Correct return types (ignoring CV)", "green"),
            ("Correct return types (strict)", "green"),
            ("Correct argument names", "green"),
            ("Correct argument types (ignoring CV)", "green"),
            ("Correct argument types (strict)", "green"),
            ("Missing arguments", "red"),
            ("Redundant arguments", "red"),
            ("Missing string literals", "red"),
            ("Redundant string literals", "red"),
            ("Pointer conversion", "green"),
        ]
        references = [[tgt] for tgt in self.references]
        references_no_var = [[tgt] for tgt in self.references_no_var]
        summary_table_cols: List[List[str]] = []
        for name, stats in self.stats.items():
            bleu4 = tx.evals.corpus_bleu(references, self.hypotheses[name], max_order=4)
            bleu4_no_var = tx.evals.corpus_bleu(references_no_var, self.hypotheses_no_var[name], max_order=4)
            bleu8 = tx.evals.corpus_bleu(references, self.hypotheses[name], max_order=8)
            summary_table_cols.append([
                name,
                f"{bleu4:.2f}",
                f"{bleu8:.2f}",
                f"{bleu4_no_var:.2f}",
                str(stats.unparsable),
                str(stats.fn_name),
                str(stats.fn_ret_type),
                str(stats.fn_ret_type_strict),
                str(stats.arg_name),
                str(stats.arg_type),
                str(stats.arg_type_strict),
                str(stats.arg_missing),
                str(stats.redundant_args),
                str(stats.missing_strings),
                str(stats.redundant_strings),
                f"P: {stats.pointer.precision}, R: {stats.pointer.recall}",
            ])
        summary_table_items: List[List[str]] = list(
            map(list, zip(*([[name for name, _ in summary_table_col_headers]] + summary_table_cols))))  # transpose
        summary_table = Markdown.Table(summary_table_items, ["left"] + ["right"] * len(summary_table_cols))
        for idx, (_, color) in enumerate(summary_table_col_headers):
            if color is not None:
                summary_table.set_color(idx, 0, color)

        improvement_tables: 'OrderedDict[str, Markdown.Table]' = OrderedDict()
        for name in self.names:
            improving_table = [["Metric", "Deteriorated (↓)", "Same (-)", "Improved (↑)"]]
            for key, group in self.stats[name].improving.group_by(lambda xs: xs[0]).items():
                values = {diff: count for (_, diff), count in group}
                total = sum(values.values())
                improving_table.append([Stats.KEY_DESCRIPTION[key]] +
                                       [f"{values.get(diff, 0)} / {total}" for diff in [-1, 0, 1]])
            improving_table = Markdown.Table(improving_table, ["left", "right", "right", "right"])
            improving_table.set_color(0, 1, "red")
            improving_table.set_color(0, 3, "green")
            improvement_tables[name] = improving_table

        print(summary_table.to_str(show_colors=True), end='\n\n')
        for name in self.names:
            stats = self.stats[name]
            print(colored(f"{name}:", "blue"))
            print(colored("  Improving:", "yellow"))
            print(Markdown.indent(improvement_tables[name].to_str(show_colors=True), indent=4))
            print(colored("  Arg name categories:", "yellow"))
            print(Markdown.indent(stats.arg_name_kind.to_string(lambda xs: xs[1]), indent=4))
            print(colored("  Arg type categories:", "yellow"))
            print(Markdown.indent(stats.arg_type_kind.to_string(lambda xs: xs[1]), indent=4))

        if self.exporter is not None:
            self.exporter.generate(self.stats, summary_table, improvement_tables)


class InputData(NamedTuple):
    names: List[str]
    src_data: List[str]
    tgt_data: List[str]
    hyp_data: Dict[str, List[str]]
    overlap_scores: Dict[str, List[float]]
    additional_data: List[Tuple[str, str, str, str]]  # (var_map, score, repo, sha)


def main():
    args = Args()
    flutes.register_ipython_excepthook()
    with open(args.test_file, "rb") as f:
        data = InputData(*pickle.load(f))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    for file_name, max_size in [("eval-small", 100), ("eval", len(data.src_data))]:
        # exporter = HTMLExporter(os.path.join(args.output_dir, file_name + ".html"), data.names)
        exporter = JSONExporter(os.path.join(args.output_dir, file_name + ".json"), data.names)
        evaluator = Evaluator(data.names, exporter=exporter)
        for idx in trange(max_size):
            var_names, score, repo, sha = data.additional_data[idx]
            var_map = {}
            if var_names != "":
                var_map = {name: (decomp, oracle) for name, decomp, oracle in flutes.chunk(3, var_names.split("\0"))}
            evaluator.add(src=data.src_data[idx],
                          tgt=data.tgt_data[idx],
                          hyps={name: data.hyp_data[name][idx] for name in data.names},
                          overlap_scores={name: data.overlap_scores[name][idx] for name in data.names},
                          var_map=var_map, repo=repo, sha=sha)
        evaluator.print_summary()


if __name__ == '__main__':
    main()
