# Party Context Cohesion V9 (2026-07-27)

## Corrected interpretation

Party-internal context is not a measure of total public support. It estimates
how much of an already aligned camp remains consolidated behind its candidate.

V8 incorrectly added centered party support and fragmentation directly to each
candidate's total prediction. This let elite context behave like nationwide
candidate approval and particularly overstated the effect of comparative party
fragmentation.

V9 replaces the direct adjustment with a mass-conserving retention mechanism:

1. Build defection risk from dated party-context support, fragmentation, and
   confidence.
2. Allow at most 2% of candidate-aligned core mass to become contestable.
3. Allow at most 15% of critical-support mass to become contestable.
4. Return released mass to the region's flexible pool and allocate it by the
   pre-adjustment prediction.
5. Normalize each election-region contest to 100%.

The former indirect path through centered coalition mobilization and conversion
capacity is also disabled. Direct candidate conversion now uses only public
stature, legitimacy, organization, and alternative-candidate evidence. Party
context remains available for supporter retention and within-bloc dispersion.

No context or zero confidence is an exact identity. The mechanism never creates
votes and never applies party context directly to the whole electorate.

## Strict nested comparison

| Metric | V8 direct adjustment | V9 cohesion | Change |
|---|---:|---:|---:|
| Regional weighted macro MAE | 3.9499%p | 3.8584%p | -0.0915%p |
| National candidate macro MAE | 2.6298%p | 2.4769%p | -0.1529%p |
| Winner accuracy | 80% | 80% | 0%p |

| Election | V8 national MAE | V9 national MAE | Change |
|---|---:|---:|---:|
| 2002 | 3.4083%p | 3.3455%p | -0.0628%p |
| 2007 | 4.8983%p | 4.4472%p | -0.4511%p |
| 2012 | 1.4952%p | 1.2717%p | -0.2235%p |
| 2017 | 3.0114%p | 3.2122%p | +0.2008%p |
| 2022 | 0.3359%p | 0.1079%p | -0.2280%p |

The caps are theory-constrained constants, not values selected by searching the
five presidential outcomes. Historical through-2022 development-sample caveats
still apply to the rest of the active stack.
