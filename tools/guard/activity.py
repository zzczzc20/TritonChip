"""Conservative, content-based code activity accounting for the watchdog."""
import ast
import hashlib
import io
import re
import token
import tokenize

try:
    from .guard import is_code
except ImportError:
    from guard import is_code

GENERATED = re.compile(r'(^|/)(?:gen|generated|build|dist|__pycache__)(?:/|$)')


def watched_code(path):
    return is_code(path) and not GENERATED.search(path)


def fingerprint(line):
    return hashlib.sha256(line.encode('utf-8', 'surrogateescape')).hexdigest()[:24]


def python_lines(text):
    ignored = set()
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            body = getattr(node, 'body', None)
            if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                ignored.update(range(body[0].lineno, body[0].end_lineno + 1))
    except (SyntaxError, ValueError):
        pass  # A work in progress may not parse; tokenization can still work.
    lines = {}
    skipped = {tokenize.COMMENT, tokenize.NL, token.NEWLINE, token.INDENT,
               token.DEDENT, token.ENDMARKER, tokenize.ENCODING}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type not in skipped and tok.start[0] not in ignored:
                lines.setdefault(tok.start[0], []).append(f'{tok.type}:{tok.string}')
        return ['|'.join(parts) for _, parts in sorted(lines.items())]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return c_like_lines(text, hash_comments=True)


def c_like_lines(text, hash_comments=False):
    lines = []
    block = False
    for line in text.splitlines():
        out = []
        quote = None
        escaped = False
        i = 0
        while i < len(line):
            c = line[i]
            following = line[i:i+2]
            if block:
                if following == '*/':
                    block = False
                    i += 2
                else:
                    i += 1
                continue
            if quote:
                out.append(c)
                if escaped:
                    escaped = False
                elif c == '\\':
                    escaped = True
                elif c == quote:
                    quote = None
                i += 1
                continue
            if following == '/*':
                block = True
                i += 2
            elif following == '//' or (hash_comments and c == '#'):
                break
            elif c in ('"', "'"):
                quote = c
                out.append(c)
                i += 1
            else:
                if not c.isspace():
                    out.append(c)
                i += 1
        if out:
            lines.append(''.join(out))
    return lines


def code_lines(path, content):
    if not watched_code(path) or b'\0' in content:
        return []
    text = content.decode('utf-8', 'surrogateescape')
    if path.endswith('.py'):
        return python_lines(text)
    return c_like_lines(text, hash_comments=path.endswith(('.sh', '.bash', '.tcl')) or path == 'Makefile')


class Novelty:
    """Count unique novel lines and first-time removals, never a toggle twice."""
    def __init__(self, baseline):
        self.previous = set(baseline)
        self.baseline = set(baseline)
        self.seen = set(baseline)
        self.removed = set()

    def update(self, current):
        current = set(current)
        added = (current - self.previous) - self.seen
        removed = (self.previous - current) & self.baseline - self.removed
        self.seen.update(current)
        self.removed.update(removed)
        changed = max(len(added), len(removed))
        self.previous = current
        return changed
