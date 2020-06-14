from typing import Iterator, List

from pycparser.c_lexer import CLexer

__all__ = [
    "LexToken",
    "Lexer",
]


class LexToken:  # stub
    type: str
    value: str
    lineno: int
    lexpos: int


class Lexer:
    @staticmethod
    def _error_func(msg, loc0, loc1):
        pass

    @staticmethod
    def _brace_func():
        pass

    @staticmethod
    def _type_lookup_func(typ):
        return False

    def __init__(self) -> None:
        self.lexer = CLexer(self._error_func, self._brace_func, self._brace_func, self._type_lookup_func)
        self.lexer.build(optimize=True, lextab='pycparser.lextab')

    def lex_tokens(self, code: str) -> Iterator[LexToken]:
        self.lexer.reset_lineno()
        self.lexer.input(code)
        while True:
            token = self.lexer.token()
            if token is None:
                break
            yield token

    def lex(self, code: str) -> List[str]:
        return [token.value for token in self.lex_tokens(code)]
