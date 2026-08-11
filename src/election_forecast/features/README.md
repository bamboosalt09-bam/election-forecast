# Features

Feature builders transform normalized source tables into model-ready columns.

Planned modules:

- `region_bloc_prior.py`: repeated proportional-election and presidential bloc structure.
- `issue_matcher.py`: keyword and phrase-level issue matching for speeches/news text.
- `issue_features.py`: issue salience, candidate issue link, and regional sensitivity.
- `candidate_features.py`: candidate general premium and latent party vectors.
- `event_features.py`: coalition, withdrawal, endorsement, and split effects.

Rules:

- Features must be computed with `available_date <= forecast_date`.
- Feature code should not hard-code political judgments; use CSV/config mappings.
- Interaction features belong here when they are reusable across model classes.

Issue matching notes:

- Keyword CSV values may contain either single words or multi-word phrases.
- Phrase terms allow whitespace, punctuation, and common Korean particles between tokens.
- Example: `청년 실업` can match `청년 실업`, `청년의 실업`, or punctuation-separated variants.
- Phrase terms can also match ordered tokens separated by other words inside the same sentence.
- The default maximum gap between phrase tokens is 40 characters to limit false positives.
- Overlapping matches are resolved once: phrase matches win before single words, shorter phrase spans win before longer overlapping phrase spans, and matched text spans are not reused.
