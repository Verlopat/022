"""Scan manuscript text for forbidden words, phrases, sentence templates and first-person pronouns."""
import re
import sys

WORDS = [
    'furthermore', 'moreover', 'additionally', 'notably', 'importantly', 'significantly', 'consequently',
    'therefore', 'hence', 'thus', 'overall', 'however', 'similarly',
    'robust', 'robustness', 'efficient', 'efficiently', 'scalable', 'scalability', 'innovative', 'innovation',
    'comprehensive', 'comprehensively', 'advanced', 'cutting-edge', 'transformative', 'novel', 'novelty',
    'paradigm', 'paradigms', 'framework', 'frameworks', 'perspective', 'perspectives', 'approach', 'approaches',
    'methodology', 'methodologies', 'strategy', 'strategies', 'mechanism', 'mechanisms', 'architecture',
    'architectures', 'implementation', 'implementations', 'implement', 'implemented', 'implementing',
    'optimization', 'optimisation', 'optimize', 'optimise', 'optimized', 'optimised', 'optimal', 'optimum',
    'optimizer', 'optimiser',
    'demonstrate', 'demonstrates', 'demonstrated', 'demonstrating', 'highlight', 'highlights', 'highlighted',
    'indicate', 'indicates', 'indicated', 'indicating', 'suggest', 'suggests', 'suggested', 'suggesting',
    'reveal', 'reveals', 'revealed', 'illustrate', 'illustrates', 'illustrated', 'emphasize', 'emphasizes',
    'emphasise', 'emphasises', 'support', 'supports', 'supported', 'supporting', 'validate', 'validates',
    'validated', 'validating',
]
PHRASES = [
    'in conclusion', 'in summary', 'it is worth noting', 'it should be noted', 'it is important to highlight',
    'on the other hand', 'in contrast', 'by contrast', 'in addition', 'for instance', 'for example',
    'as a result', 'provides insight', 'provide insight', 'overall results', 'future research',
    'potential implications', 'key findings', 'significant improvement', 'promising solution', 'important role',
    'critical importance', 'this study aims', 'this paper explores', 'this research investigates',
    'the findings suggest', 'the results demonstrate', 'in recent years', 'increasing interest',
    'the results indicate', 'this research highlights', 'the findings provide', 'valuable insights',
    'a comprehensive analysis', 'this approach offers', 'should focus on',
]
PRONOUNS = ['we', 'our', 'ours', 'us', 'i', 'my', 'me', 'myself', 'ourselves']
SOFT = ['significant', 'efficiency', 'validation', 'crucial', 'pivotal', 'leverage', 'leverages', 'utilize',
        'utilizes', 'delve', 'seamless', 'seamlessly', 'holistic', 'state-of-the-art', 'notable']


def check(text, label=''):
    issues = []
    low = text.lower()
    for w in WORDS:
        for m in re.finditer(r'(?<![a-z\-])' + re.escape(w) + r'(?![a-z])', low):
            ctx = text[max(0, m.start() - 40):m.end() + 40].replace('\n', ' ')
            issues.append(('WORD', w, ctx))
    for p in PHRASES:
        for m in re.finditer(re.escape(p), low):
            ctx = text[max(0, m.start() - 40):m.end() + 40].replace('\n', ' ')
            issues.append(('PHRASE', p, ctx))
    for w in PRONOUNS:
        pat = r'(?<![A-Za-z_\\$\-])' + (w if w != 'i' else 'I') + r'(?![A-Za-z_\-$])'
        flags = 0 if w == 'i' else re.IGNORECASE
        for m in re.finditer(pat, text, flags):
            ctx = text[max(0, m.start() - 40):m.end() + 40].replace('\n', ' ')
            issues.append(('PRONOUN', w, ctx))
    soft = []
    for w in SOFT:
        for m in re.finditer(r'(?<![a-z])' + re.escape(w) + r'(?![a-z])', low):
            soft.append(w)
    return issues, soft


if __name__ == '__main__':
    txt = open(sys.argv[1]).read()
    if '# References' in txt:
        txt = txt.split('# References')[0]
    issues, soft = check(txt)
    for it in issues:
        print(it)
    print('HARD ISSUES:', len(issues))
    from collections import Counter
    print('soft words:', Counter(soft))
