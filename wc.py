import re, sys
def count(t):
    t = re.sub(r'\$\$.*?\$\$', ' ', t, flags=re.S)
    t = re.sub(r'\$[^$]*\$', ' X ', t)
    t = re.sub(r'^(TABLE|ALGORITHM|FIGURE):.*$', '', t, flags=re.M)
    t = re.sub(r'\[@\d+\]', '', t)
    t = re.sub(r'^#+ .*$', '', t, flags=re.M)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-\.,%/]*", t))
for f in sys.argv[1:]:
    print(f, count(open(f).read()))
