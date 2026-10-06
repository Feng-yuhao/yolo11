# D44: frozen routing / Dense inference interventions

ZERO TRAINING. No checkpoint/source overwrite. No automatic shutdown.
Off-manifold sensitivity, not training-time causal proof or a new trained model.

|Model|Mode|AP|AP50|AP75|AR500|AP delta vs SAME checkpoint normal|
|---|---|---:|---:|---:|---:|---:|
|e40|normal|26.222452|45.361316|25.844454|38.060303|+0.000000|
|e40|uniform_depth|24.784577|43.350646|23.904149|37.398555|-1.437876|
|e40|shared_task|26.214075|45.334990|25.833730|38.065488|-0.008378|
|e40s2|normal|26.063860|45.213971|25.664076|37.897175|+0.000000|
|e40s2|uniform_depth|24.736325|43.788356|23.819454|37.352876|-1.327535|
|e40s2|shared_task|25.992634|45.075824|25.460850|37.849238|-0.071226|
|e42|normal|26.012438|44.927822|25.516130|38.227862|+0.000000|
|e42|channel_off_p2|26.005416|44.919021|25.502411|38.233306|-0.007021|
|e42|channel_off_p3|26.013185|44.935999|25.519390|38.229829|+0.000747|
|e43|normal|26.157557|45.305637|25.720958|38.158174|+0.000000|
|e43|dense_bypass_p2|21.027358|37.084187|20.035145|34.788039|-5.130199|
|e43|dense_bypass_p3|26.066665|45.207168|25.647549|38.116154|-0.090892|

## Interpretation boundaries

- Compare intervention with its OWN checkpoint default, not E40 with different trained weights.
- Turning an operation off at validation is not its matched retraining ablation.
- A rescue supports a follow-up hypothesis; a drop can be distribution mismatch/co-adaptation.
- Small AP differences can be numerical/seed effects; do not declare a root cause.
- Do not select the best intervention as a final paper model or automatically launch E44/E45.
- routing_rows.json uses GT centers / 3x3 feature neighborhoods. Entropy/JS are descriptive.
- Input sqrt(area) groups are NOT official small-object AP. No test-dev is used.

## Directional hints (not causal conclusions)

```json
{
  "E44": {
    "frozen_rescue_at_levels": [
      "P3"
    ],
    "boundary": "Not proof that S+C competition caused training degradation; independent compensation still needs controlled training."
  },
  "E45": {
    "observed_P2_bypass_rescue_and_P3_bypass_not_rescue": false,
    "boundary": "Even this pattern does not isolate training causality or guarantee P3-only Dense will help."
  },
  "E40_uniform_depth": {
    "AP_delta_seed1": -1.4378756799593972,
    "AP_delta_seed2": -1.3275350256672773,
    "both_negative": true,
    "boundary": "Measures trained model dependence, not which component originally created the training gain."
  },
  "E40_shared_task": {
    "AP_delta_seed1": -0.008377559825998304,
    "AP_delta_seed2": -0.07122636902570889,
    "both_negative": true,
    "boundary": "Measures trained model dependence, not which component originally created the training gain."
  }
}
```
