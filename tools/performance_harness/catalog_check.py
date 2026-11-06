#!/usr/bin/env python3
"""Coverage check of the Performance Harness tool catalog (tools/performance_harness/README.md).

    python3 tools/performance_harness/catalog_check.py [--root REPO] [--readme FILE] [--list]

Scans the tool directories (SCAN_DIRS) for scripts (.py, .sh, .tcl) and fails when
  * a script is neither listed in the README catalog nor matched by IGNORE, or
  * a path or glob listed in the README catalog no longer exists.

The catalog is the region of the README between the lines
`<!-- BEGIN catalog -->` and `<!-- END catalog -->`. A script is listed when a backticked
token in that region equals its repository-relative path or is a glob that matches it
(`*` does not cross `/`). Backticked tokens elsewhere in the README that look like a script
path under SCAN_DIRS must also exist.

It only reads files, imports nothing outside the standard library and never runs a tool.
Exit status: 0 complete, 1 coverage or stale-path findings, 2 usage or README error.
"""
import argparse
import os
import re
import sys

SCAN_DIRS = (
    'tools/performance_harness',
    'tools/r41',
    'archbench',
    'tools/models',
    'tools/model',
    'tools/isa',
)
SUFFIXES = ('.py', '.sh', '.tcl')
# Scripts that are deliberately not catalogued, as exact repository-relative paths or globs.
IGNORE = (
    'archbench/__init__.py',
    'archbench/platforms/__init__.py',
)
SKIP_DIRNAMES = ('__pycache__',)
BEGIN = '<!-- BEGIN catalog -->'
END = '<!-- END catalog -->'
TOKEN = re.compile(r'`([^`\s]+)`')


def glob_regex(pattern):
    out = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == '*':
            out.append('[^/]*')
        elif c == '?':
            out.append('[^/]')
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile('^' + ''.join(out) + '$')


def is_glob(token):
    return any(c in token for c in '*?')


def scan(root):
    found = []
    for base in SCAN_DIRS:
        top = os.path.join(root, base)
        for dirpath, dirnames, filenames in os.walk(top):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRNAMES)
            for name in sorted(filenames):
                if name.endswith(SUFFIXES):
                    rel = os.path.relpath(os.path.join(dirpath, name), root)
                    found.append(rel.replace(os.sep, '/'))
    return sorted(found)


def split_readme(text):
    lines = text.split('\n')
    try:
        b = next(i for i, l in enumerate(lines) if l.strip() == BEGIN)
        e = next(i for i, l in enumerate(lines) if l.strip() == END)
    except StopIteration:
        raise ValueError('README has no %s ... %s region' % (BEGIN, END))
    if e < b:
        raise ValueError('catalog END marker precedes BEGIN')
    return '\n'.join(lines[b + 1:e]), '\n'.join(lines[:b] + lines[e + 1:])


def path_like(token):
    return token.endswith(SUFFIXES) and any(token.startswith(d + '/') for d in SCAN_DIRS)


def check(root, readme_text, ignore=IGNORE):
    region, rest = split_readme(readme_text)
    catalog_tokens = sorted(set(t for t in TOKEN.findall(region) if path_like(t)))
    other_tokens = sorted(set(t for t in TOKEN.findall(rest) if path_like(t) and not is_glob(t)))
    scripts = scan(root)
    matchers = [(t, glob_regex(t)) for t in catalog_tokens]
    ignorers = [glob_regex(p) for p in ignore]
    uncovered = [s for s in scripts
                 if not any(m.match(s) for _, m in matchers) and not any(m.match(s) for m in ignorers)]
    stale = []
    for token, m in matchers:
        if is_glob(token):
            if not any(m.match(s) for s in scripts):
                stale.append(token)
        elif not os.path.isfile(os.path.join(root, token)):
            stale.append(token)
    stale_other = [t for t in other_tokens if not os.path.isfile(os.path.join(root, t))]
    return {
        'scripts': len(scripts),
        'catalog_entries': len(catalog_tokens),
        'uncovered': uncovered,
        'stale_catalog_paths': stale,
        'stale_other_paths': stale_other,
    }


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=os.path.abspath(os.path.join(here, '..', '..')))
    ap.add_argument('--readme', default=None, help='default: ROOT/tools/performance_harness/README.md')
    ap.add_argument('--list', action='store_true', help='print every scanned script')
    args = ap.parse_args(argv)
    readme = args.readme or os.path.join(args.root, 'tools/performance_harness/README.md')
    try:
        with open(readme, encoding='utf-8') as f:
            text = f.read()
        result = check(args.root, text)
    except (OSError, ValueError) as exc:
        print('catalog_check: %s' % exc, file=sys.stderr)
        return 2
    if args.list:
        print('\n'.join(scan(args.root)))
    bad = False
    for key, title in (('uncovered', 'script not in the README catalog and not in IGNORE'),
                       ('stale_catalog_paths', 'catalog path or glob that matches nothing'),
                       ('stale_other_paths', 'README path outside the catalog that does not exist')):
        for item in result[key]:
            print('%s: %s' % (title, item))
            bad = True
    print('catalog_check: %d scripts, %d catalog entries, %d uncovered, %d stale'
          % (result['scripts'], result['catalog_entries'], len(result['uncovered']),
             len(result['stale_catalog_paths']) + len(result['stale_other_paths'])))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
