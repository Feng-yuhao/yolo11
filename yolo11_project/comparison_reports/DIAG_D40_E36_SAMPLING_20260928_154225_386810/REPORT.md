# D40 E36 frozen sampling audit

Inference interventions can change feature distributions. They do not prove a retraining outcome.

|Arm/mode|Official AP|AP50|AP75|
|---|---:|---:|---:|
|E36a/normal|25.6111|44.5976|24.9678|
|E36a/mask_off|23.1817|40.9886|22.4262|
|E36b/normal|25.5162|44.2100|25.0162|
|E36b/offset_off|19.4795|37.7790|16.4235|
|E36b/mask_off|22.6833|40.1920|22.0462|
|E36b/both_off|21.2372|40.1198|19.3750|

|Arm/mode/size|GT|raw any .75|P2 any .75|raw correct .75|top500 .75|
|---|---:|---:|---:|---:|---:|
|E36a/normal/all|38599|62.639|56.416|55.631|46.900|
|E36a/normal/lt8|11819|26.347|24.410|21.618|17.455|
|E36a/normal/8_16|15114|68.321|65.668|60.050|47.737|
|E36a/normal/16_24|6186|88.458|85.483|80.068|69.609|
|E36a/normal/24_32|2548|94.270|88.148|87.049|79.239|
|E36a/normal/ge32|2932|97.681|48.840|91.098|85.266|
|E36a/mask_off/all|38599|59.958|53.672|53.069|43.771|
|E36a/mask_off/lt8|11819|19.029|15.940|15.069|11.295|
|E36a/mask_off/8_16|15114|66.614|63.524|58.158|44.627|
|E36a/mask_off/16_24|6186|89.719|87.827|81.005|69.560|
|E36a/mask_off/24_32|2548|94.466|89.874|87.441|79.003|
|E36a/mask_off/ge32|2932|97.851|51.467|91.201|85.232|
|E36b/normal/all|38599|62.618|57.781|55.789|46.973|
|E36b/normal/lt8|11819|25.772|23.691|21.102|17.108|
|E36b/normal/8_16|15114|68.466|65.893|60.368|47.870|
|E36b/normal/16_24|6186|88.749|86.421|80.585|69.673|
|E36b/normal/24_32|2548|94.819|90.424|88.187|80.298|
|E36b/normal/ge32|2932|97.885|64.598|91.542|85.880|
|E36b/offset_off/all|38599|62.336|57.325|55.279|42.208|
|E36b/offset_off/lt8|11819|25.611|23.598|20.856|16.287|
|E36b/offset_off/8_16|15114|68.063|64.933|59.508|42.484|
|E36b/offset_off/16_24|6186|88.280|85.742|80.052|58.147|
|E36b/offset_off/24_32|2548|94.819|90.345|87.951|72.057|
|E36b/offset_off/ge32|2932|97.885|65.416|91.576|85.709|
|E36b/mask_off/all|38599|61.432|56.463|54.457|44.866|
|E36b/mask_off/lt8|11819|21.821|19.232|17.506|13.402|
|E36b/mask_off/8_16|15114|68.089|65.191|59.528|45.752|
|E36b/mask_off/16_24|6186|89.735|88.070|81.102|69.690|
|E36b/mask_off/24_32|2548|94.976|90.306|88.422|78.493|
|E36b/mask_off/ge32|2932|97.920|65.450|91.542|85.539|
|E36b/both_off/all|38599|61.230|56.284|54.203|42.294|
|E36b/both_off/lt8|11819|21.499|18.792|17.100|12.421|
|E36b/both_off/8_16|15114|68.043|65.112|59.475|43.304|
|E36b/both_off/16_24|6186|89.169|87.100|80.424|62.464|
|E36b/both_off/24_32|2548|95.055|90.738|88.344|75.903|
|E36b/both_off/ge32|2932|97.920|66.951|91.610|85.744|

Baseline D38 slices and per-GT gained/lost counts are included in each metrics JSON.
Sampling statistics describe input-conditioned outputs, not parameter magnitudes.
Background excludes raw GT rectangles (including ignore regions) and letterbox padding.
Coverage is candidate existence, not one-to-one recall or AP.
Statistics near targets use a 3x3 P2-cell center neighborhood; overlaps are not causal attribution.
No automatic new experiment or threshold tuning is authorized by this report.
