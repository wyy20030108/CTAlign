"""Frozen English report-wording strata, not adjudicated pleural effusion status."""
import re

VERSION = 'report-wording-v1'
TARGET = re.compile(
    r'\b(?:pleural\s+(?:(?:space|fluid|and|or|pericardial|free|loculated)\s+){0,3}(?:effusions?|fluid|collection)|'
    r'(?:fluid|effusions?)\b[^.;!?]{0,55}?\b(?:pleural\s+(?:spaces?|cavit(?:y|ies))|hemithora(?:x|ces))|'
    r'hydrothorax)\b', re.I)
NEG_BEFORE = re.compile(r'\b(?:no|without|negative for|absence of|free of)\b', re.I)
NEG_AFTER = re.compile(r'\b(?:absent|not (?:present|observed|detected|seen|identified|found|noted|evident|visualized))\b', re.I)
UNCERTAIN = re.compile(r'\b(?:cannot (?:exclude|rule out|be excluded|be ruled out)|not excluded|'
                       r'questionable|possibly|possible|may (?:be|represent)|might|suspected|suspicious for|versus|vs\.?)\b|\?', re.I)
PAST = re.compile(r'\b(?:resolved|no longer (?:seen|present)|history of|previously (?:seen|noted|present)|'
                  r'prior (?:finding of|study showed))\b', re.I)
CURRENT = re.compile(r'\b(?:persist\w*|remain\w*|unchanged|stable|increased|decreased|newly|interval increase|interval decrease)\b',re.I)
SPLIT = re.compile(r'[;!\n]+|(?<!\d)\.(?!\d)|\.(?=\s+[A-Za-z])|\b(?:however|whereas|but)\b', re.I)
SIZE_GROUPS = {'small':r'\b(?:trace|minimal|small|mild|slight)\b','moderate':r'\bmoderate\b','large':r'\b(?:large|massive)\b'}


def laterality(text):
    bilateral = re.search(r'\b(?:bilateral(?:ly)?|both (?:sides|pleural|hemithoraces|lungs))\b',text,re.I)
    words = set(re.findall(r'\b(?:left|right)\b',text.lower()))
    if bilateral: return 'bilateral'
    if len(words)==2: return 'both_sides_mentioned'
    return next(iter(words),'unspecified')


def describe_report(findings, impressions):
    statuses, sides, sizes = set(),set(),set()
    ambiguous = False
    for field in (findings or '', impressions or ''):
        for clause in SPLIT.split(field):
            hits=list(TARGET.finditer(clause))
            if not hits:
                # A generic effusion can be pericardial, articular or anatomically unspecified.
                if re.search(r'\beffusions?\b',clause,re.I) and not re.search(r'\b(?:pericardial|joint|articular)\b',clause,re.I):
                    ambiguous=True
                continue
            for m in hits:
                before=clause[max(0,m.start()-75):m.start()]
                after=clause[m.end():m.end()+75]
                nearby=before+clause[m.start():m.end()]+after
                # Limit cues from unrelated predicates, preserving coordinated "no X or pleural effusion".
                before=re.split(r'\b(?:there (?:is|are)|demonstrates?|reveals?|shows?)\b',before,flags=re.I)[-1]
                if UNCERTAIN.search(nearby): state='uncertain'
                elif PAST.search(nearby) and not CURRENT.search(nearby): state='historical'
                elif NEG_BEFORE.search(before) or NEG_AFTER.search(after): state='negated'
                else: state='affirmed'
                statuses.add(state)
                if state=='affirmed':
                    sides.add(laterality(nearby))
                    sizes.update(k for k,pattern in SIZE_GROUPS.items() if re.search(pattern,nearby,re.I))
    if 'affirmed' in statuses and ('negated' in statuses or 'uncertain' in statuses): state='conflict'
    elif 'affirmed' in statuses: state='affirmed'
    elif 'uncertain' in statuses: state='uncertain'
    elif 'negated' in statuses: state='negated'
    elif 'historical' in statuses: state='historical'
    else: state='uncertain' if ambiguous else 'unmentioned'
    stratum={'affirmed':'affirmed','negated':'explicitly_negated'}.get(state,'uncertain_or_unmentioned')
    sides.discard('unspecified')
    side=next(iter(sides)) if len(sides)==1 else ('multiple_or_discordant' if sides else 'unspecified')
    size=next(iter(sizes)) if len(sizes)==1 else ('multiple' if sizes else 'unspecified')
    return {'state':state,'stratum':stratum,'side':side,'size':size}
