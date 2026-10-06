# D41 frozen-feature paired head probe

| model | Official AP | AP50 | AP75 | raw coverage .75 |
|---|---:|---:|---:|---:|
|baseline|25.6312|44.7255|25.0298|0.0259|
|control|25.5671|44.7114|24.8896|0.0233|
|mask|25.5678|44.7097|24.8939|0.0233|

Mask - synchronized control AP: +0.0008 pp; AP75: +0.0043 pp.
Predeclared screening signal: False.

Fixed endpoint, seed1, 10 epochs; no EMA or best-checkpoint selection.
BN running statistics and all E1 parameters outside the two copied P2 regression towers stay fixed.
Standard TAL uses each arm's current boxes, so assignments can differ despite identical TAL code.
This tests head utility on fixed E1 features. It does not identify a unique root cause or prove a full-training outcome.
Raw coverage is independent candidate existence, not one-to-one recall; size bins use 640-input pixels, not COCO original-image area.
