"""Human curation of the auto-recovered label routings (EDIT ME freely).

Context: relabel_map.py auto-routes wrongly-dropped words to their nearest wheel
canonical by BGE cosine. That first draft is ~40% correct but (a) forces genuinely
MISSING concepts (humor, neutral, serious) into wrong buckets, and (b) recovers
non-emotion words the auto-annotator emitted ("advice", "professional", "objective").
This file overrides the auto map with reviewed decisions (policy 2026-06-30:
add humor+neutral+serious canonicals; drop only CLEAR non-emotions).

Two structures, both editable by hand:
  NEW_CANONICALS : concept missing from the wheel -> the vocab words that belong to
                   it (these override whatever the wheel/auto map said).
  DROP           : auto-recovered words that are NOT emotions (or wrong valence) and
                   should stay dropped (not scorable).
Everything else that was auto-recovered keeps its auto routing.
"""

# New canonical emotions the OV-MER wheel lacks (note for advisor: these extend
# the wheel's label space). Member words are pulled from the vocab regardless of
# frequency, so the concept is recovered even where the wheel dropped it.
NEW_CANONICALS = {
    "humorous": ["humorous", "humor", "humour", "joking", "jokey", "jest",
                 "joke", "self-mockery", "playful", "playfulness", "witty",
                 "amusing", "lighthearted", "light-hearted", "teasing"],
    "neutral":  ["neutral", "neutrality", "indifferent", "matter-of-fact"],
    "serious":  ["solemn", "solemnity", "stern", "formal", "firm", "firmness",
                 "grave", "serious", "seriousness", "sober", "earnest"],
}

# Clear non-emotions / wrong-valence routings the auto map produced -> keep dropped.
# (Kept borderline-affective words: warning, longing, cool, complaint, discomfort,
#  commitment, decisive, energy, celebration, liking — those route acceptably.)
DROP = {
    # not emotions — task / cognition / manner words
    "advice", "request", "inquiry", "professional", "professionalism",
    "control", "authority", "authoritative", "objective", "rational",
    "explanation", "emphasis", "difficulty", "resolve", "willingness",
    "willing", "exaggerated", "dramatic", "resilience", "persuasive",
    "inviting", "accusation", "humility", "self-reflection",
    # vague meta-sentiment fillers (too generic to be one emotion)
    "positive evaluation", "negative evaluation", "negative assessment",
    "positive emotions", "positive sentiment", "positive emotion",
    "negative emotions", "negative emotion", "positive mood",
    "positive attitude",
}
