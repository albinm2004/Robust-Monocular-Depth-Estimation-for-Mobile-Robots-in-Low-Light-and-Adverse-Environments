# Results Summary

Aggregated from 40248 rows across 4 CSV(s): data/results/eval_full654_small.csv, data/results/eval_full654_small_zerodcetrained_relabeled.csv, data/results/eval_subset80_base.csv, data/results/eval_subset80_large.csv

Near-field columns use the RealSense D435i's configured obstacle-detection band (0.25-0.70m); 'n/a' means this condition has zero ground-truth pixels in that band in the underlying data (see README known-simplifications).

Notes:
- `base`/`large` rows are the 80-image subset (accuracy-vs-latency comparison); `small` rows are the full 654-image test split.
- `zero_dce` = untrained (random-init) Zero-DCE, the code's default. `zero_dce_trained` = after 5000 steps of real unsupervised training (see README). Trained Zero-DCE only helps on `low_light`; it is neutral-to-harmful elsewhere because it brightens every input toward a fixed target exposure regardless of whether the input needs it (confirmed by direct inspection: it amplifies sensor noise ~10x on severe low-light inputs, since the curve formula has no denoising term).

## Accuracy vs. Latency (model size), dev GPU (RTX 4060 Laptop, CUDA 12.8)

Latency is independent of corruption/enhancement (same forward-pass cost regardless of input content), so it's reported once per model size rather than duplicated per row above. Clean-image Abs Rel is repeated here for convenience.

| Model | Clean Abs Rel (n=80, same subset) | Median latency (ms) | Mean latency (ms) | Mean FPS |
|---|---|---|---|---|
| small | 0.1140 | 52.3 | 85.1 | 11.7 |
| base  | 0.1090 | 107.3 | 152.9 | 6.5 |
| large | 0.0891 | 343.6 | 365.2 | 2.7 |

Jetson Orin Nano numbers (the actual Sherpa RP deployment target) are a separate later step per `docs/GPU_JETSON_SETUP.md` -- these are dev-workstation numbers only.

| Model | Corruption | Severity | Enhancement | n | Abs Rel | RMSE (m) | delta1 | Near-field Abs Rel | Near-field RMSE (m) | Near-field delta1 | Near-field px |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | clean | 0 | clahe | 80 | 0.1095 | 0.5316 | 0.8731 | n/a | n/a | n/a | 0 |
| base | clean | 0 | none | 80 | 0.1080 | 0.5409 | 0.8775 | n/a | n/a | n/a | 0 |
| base | clean | 0 | zero_dce | 80 | 0.1094 | 0.5452 | 0.8758 | n/a | n/a | n/a | 0 |
| base | blur | 1 | clahe | 80 | 0.1110 | 0.5357 | 0.8712 | n/a | n/a | n/a | 0 |
| base | blur | 1 | none | 80 | 0.1080 | 0.5396 | 0.8768 | n/a | n/a | n/a | 0 |
| base | blur | 1 | zero_dce | 80 | 0.1106 | 0.5502 | 0.8736 | n/a | n/a | n/a | 0 |
| base | blur | 3 | clahe | 80 | 0.1132 | 0.5222 | 0.8660 | n/a | n/a | n/a | 0 |
| base | blur | 3 | none | 80 | 0.1089 | 0.5268 | 0.8747 | n/a | n/a | n/a | 0 |
| base | blur | 3 | zero_dce | 80 | 0.1131 | 0.5466 | 0.8680 | n/a | n/a | n/a | 0 |
| base | blur | 5 | clahe | 80 | 0.1248 | 0.5274 | 0.8513 | n/a | n/a | n/a | 0 |
| base | blur | 5 | none | 80 | 0.1071 | 0.4772 | 0.8833 | n/a | n/a | n/a | 0 |
| base | blur | 5 | zero_dce | 80 | 0.1140 | 0.5053 | 0.8704 | n/a | n/a | n/a | 0 |
| base | indoor_haze | 1 | clahe | 80 | 0.1524 | 0.6800 | 0.7988 | n/a | n/a | n/a | 0 |
| base | indoor_haze | 1 | none | 80 | 0.1581 | 0.6992 | 0.7906 | n/a | n/a | n/a | 0 |
| base | indoor_haze | 1 | zero_dce | 80 | 0.1683 | 0.7290 | 0.7879 | n/a | n/a | n/a | 0 |
| base | indoor_haze | 3 | clahe | 80 | 0.2189 | 0.8494 | 0.7006 | n/a | n/a | n/a | 0 |
| base | indoor_haze | 3 | none | 80 | 0.2161 | 0.8583 | 0.6982 | n/a | n/a | n/a | 0 |
| base | indoor_haze | 3 | zero_dce | 80 | 0.2554 | 0.9218 | 0.6489 | n/a | n/a | n/a | 0 |
| base | indoor_haze | 5 | clahe | 80 | 0.2434 | 0.9208 | 0.6502 | n/a | n/a | n/a | 0 |
| base | indoor_haze | 5 | none | 80 | 0.2474 | 0.9321 | 0.6512 | n/a | n/a | n/a | 0 |
| base | indoor_haze | 5 | zero_dce | 80 | 0.2709 | 0.9378 | 0.6175 | n/a | n/a | n/a | 0 |
| base | low_light | 1 | clahe | 80 | 0.1228 | 0.5768 | 0.8545 | n/a | n/a | n/a | 0 |
| base | low_light | 1 | none | 80 | 0.1205 | 0.5789 | 0.8607 | n/a | n/a | n/a | 0 |
| base | low_light | 1 | zero_dce | 80 | 0.1234 | 0.5966 | 0.8580 | n/a | n/a | n/a | 0 |
| base | low_light | 3 | clahe | 80 | 0.2074 | 0.7324 | 0.7023 | n/a | n/a | n/a | 0 |
| base | low_light | 3 | none | 80 | 0.2084 | 0.7362 | 0.7029 | n/a | n/a | n/a | 0 |
| base | low_light | 3 | zero_dce | 80 | 0.2089 | 0.7464 | 0.6989 | n/a | n/a | n/a | 0 |
| base | low_light | 5 | clahe | 80 | 0.2917 | 0.9646 | 0.5628 | n/a | n/a | n/a | 0 |
| base | low_light | 5 | none | 80 | 0.2958 | 0.9728 | 0.5468 | n/a | n/a | n/a | 0 |
| base | low_light | 5 | zero_dce | 80 | 0.2950 | 0.9710 | 0.5500 | n/a | n/a | n/a | 0 |
| base | sensor_noise | 1 | clahe | 80 | 0.1143 | 0.5488 | 0.8682 | n/a | n/a | n/a | 0 |
| base | sensor_noise | 1 | none | 80 | 0.1113 | 0.5531 | 0.8720 | n/a | n/a | n/a | 0 |
| base | sensor_noise | 1 | zero_dce | 80 | 0.1122 | 0.5570 | 0.8716 | n/a | n/a | n/a | 0 |
| base | sensor_noise | 3 | clahe | 80 | 0.1205 | 0.5616 | 0.8601 | n/a | n/a | n/a | 0 |
| base | sensor_noise | 3 | none | 80 | 0.1177 | 0.5682 | 0.8649 | n/a | n/a | n/a | 0 |
| base | sensor_noise | 3 | zero_dce | 80 | 0.1172 | 0.5675 | 0.8662 | n/a | n/a | n/a | 0 |
| base | sensor_noise | 5 | clahe | 80 | 0.1242 | 0.5595 | 0.8520 | n/a | n/a | n/a | 0 |
| base | sensor_noise | 5 | none | 80 | 0.1204 | 0.5723 | 0.8623 | n/a | n/a | n/a | 0 |
| base | sensor_noise | 5 | zero_dce | 80 | 0.1203 | 0.5726 | 0.8597 | n/a | n/a | n/a | 0 |
| large | clean | 0 | clahe | 80 | 0.0916 | 0.5064 | 0.9094 | n/a | n/a | n/a | 0 |
| large | clean | 0 | none | 80 | 0.0891 | 0.5115 | 0.9120 | n/a | n/a | n/a | 0 |
| large | clean | 0 | zero_dce | 80 | 0.0907 | 0.5186 | 0.9091 | n/a | n/a | n/a | 0 |
| large | blur | 1 | clahe | 80 | 0.0935 | 0.5120 | 0.9066 | n/a | n/a | n/a | 0 |
| large | blur | 1 | none | 80 | 0.0896 | 0.5136 | 0.9107 | n/a | n/a | n/a | 0 |
| large | blur | 1 | zero_dce | 80 | 0.0916 | 0.5242 | 0.9078 | n/a | n/a | n/a | 0 |
| large | blur | 3 | clahe | 80 | 0.0908 | 0.4790 | 0.9084 | n/a | n/a | n/a | 0 |
| large | blur | 3 | none | 80 | 0.0884 | 0.4875 | 0.9112 | n/a | n/a | n/a | 0 |
| large | blur | 3 | zero_dce | 80 | 0.0890 | 0.4904 | 0.9089 | n/a | n/a | n/a | 0 |
| large | blur | 5 | clahe | 80 | 0.0927 | 0.4658 | 0.9033 | n/a | n/a | n/a | 0 |
| large | blur | 5 | none | 80 | 0.0863 | 0.4446 | 0.9102 | n/a | n/a | n/a | 0 |
| large | blur | 5 | zero_dce | 80 | 0.0869 | 0.4430 | 0.9087 | n/a | n/a | n/a | 0 |
| large | indoor_haze | 1 | clahe | 80 | 0.1112 | 0.5268 | 0.8673 | n/a | n/a | n/a | 0 |
| large | indoor_haze | 1 | none | 80 | 0.1140 | 0.5344 | 0.8659 | n/a | n/a | n/a | 0 |
| large | indoor_haze | 1 | zero_dce | 80 | 0.1914 | 0.7607 | 0.7548 | n/a | n/a | n/a | 0 |
| large | indoor_haze | 3 | clahe | 80 | 0.1660 | 0.7119 | 0.7636 | n/a | n/a | n/a | 0 |
| large | indoor_haze | 3 | none | 80 | 0.1726 | 0.7450 | 0.7580 | n/a | n/a | n/a | 0 |
| large | indoor_haze | 3 | zero_dce | 80 | 0.2558 | 0.9065 | 0.6384 | n/a | n/a | n/a | 0 |
| large | indoor_haze | 5 | clahe | 80 | 0.2053 | 0.8272 | 0.7116 | n/a | n/a | n/a | 0 |
| large | indoor_haze | 5 | none | 80 | 0.2245 | 0.8769 | 0.6744 | n/a | n/a | n/a | 0 |
| large | indoor_haze | 5 | zero_dce | 80 | 0.2712 | 0.9289 | 0.6052 | n/a | n/a | n/a | 0 |
| large | low_light | 1 | clahe | 80 | 0.1033 | 0.5375 | 0.8865 | n/a | n/a | n/a | 0 |
| large | low_light | 1 | none | 80 | 0.1017 | 0.5476 | 0.8898 | n/a | n/a | n/a | 0 |
| large | low_light | 1 | zero_dce | 80 | 0.1043 | 0.5596 | 0.8838 | n/a | n/a | n/a | 0 |
| large | low_light | 3 | clahe | 80 | 0.1940 | 0.7401 | 0.7247 | n/a | n/a | n/a | 0 |
| large | low_light | 3 | none | 80 | 0.1853 | 0.7208 | 0.7371 | n/a | n/a | n/a | 0 |
| large | low_light | 3 | zero_dce | 80 | 0.1856 | 0.7155 | 0.7369 | n/a | n/a | n/a | 0 |
| large | low_light | 5 | clahe | 80 | 0.2995 | 0.9783 | 0.5198 | n/a | n/a | n/a | 0 |
| large | low_light | 5 | none | 80 | 0.3103 | 1.0049 | 0.4911 | n/a | n/a | n/a | 0 |
| large | low_light | 5 | zero_dce | 80 | 0.3092 | 1.0019 | 0.4939 | n/a | n/a | n/a | 0 |
| large | sensor_noise | 1 | clahe | 80 | 0.0965 | 0.5195 | 0.8997 | n/a | n/a | n/a | 0 |
| large | sensor_noise | 1 | none | 80 | 0.0926 | 0.5214 | 0.9030 | n/a | n/a | n/a | 0 |
| large | sensor_noise | 1 | zero_dce | 80 | 0.0932 | 0.5230 | 0.9021 | n/a | n/a | n/a | 0 |
| large | sensor_noise | 3 | clahe | 80 | 0.0990 | 0.5195 | 0.8972 | n/a | n/a | n/a | 0 |
| large | sensor_noise | 3 | none | 80 | 0.0961 | 0.5275 | 0.9019 | n/a | n/a | n/a | 0 |
| large | sensor_noise | 3 | zero_dce | 80 | 0.0965 | 0.5266 | 0.8993 | n/a | n/a | n/a | 0 |
| large | sensor_noise | 5 | clahe | 80 | 0.1048 | 0.5273 | 0.8818 | n/a | n/a | n/a | 0 |
| large | sensor_noise | 5 | none | 80 | 0.1003 | 0.5324 | 0.8954 | n/a | n/a | n/a | 0 |
| large | sensor_noise | 5 | zero_dce | 80 | 0.1011 | 0.5335 | 0.8932 | n/a | n/a | n/a | 0 |
| small | clean | 0 | clahe | 654 | 0.1080 | 0.4600 | 0.8776 | n/a | n/a | n/a | 0 |
| small | clean | 0 | none | 654 | 0.1047 | 0.4550 | 0.8823 | n/a | n/a | n/a | 0 |
| small | clean | 0 | zero_dce | 654 | 0.1108 | 0.4744 | 0.8706 | n/a | n/a | n/a | 0 |
| small | clean | 0 | zero_dce_trained | 654 | 0.1904 | 0.6518 | 0.7333 | n/a | n/a | n/a | 0 |
| small | blur | 1 | clahe | 654 | 0.1099 | 0.4619 | 0.8738 | n/a | n/a | n/a | 0 |
| small | blur | 1 | none | 654 | 0.1054 | 0.4523 | 0.8812 | n/a | n/a | n/a | 0 |
| small | blur | 1 | zero_dce | 654 | 0.1139 | 0.4779 | 0.8658 | n/a | n/a | n/a | 0 |
| small | blur | 1 | zero_dce_trained | 654 | 0.2085 | 0.6934 | 0.7052 | n/a | n/a | n/a | 0 |
| small | blur | 3 | clahe | 654 | 0.1439 | 0.5338 | 0.8151 | n/a | n/a | n/a | 0 |
| small | blur | 3 | none | 654 | 0.1225 | 0.4768 | 0.8541 | n/a | n/a | n/a | 0 |
| small | blur | 3 | zero_dce | 654 | 0.1734 | 0.6085 | 0.7651 | n/a | n/a | n/a | 0 |
| small | blur | 3 | zero_dce_trained | 654 | 0.2953 | 0.9025 | 0.5764 | n/a | n/a | n/a | 0 |
| small | blur | 5 | clahe | 654 | 0.2358 | 0.7616 | 0.6664 | n/a | n/a | n/a | 0 |
| small | blur | 5 | none | 654 | 0.1771 | 0.6164 | 0.7610 | n/a | n/a | n/a | 0 |
| small | blur | 5 | zero_dce | 654 | 0.3080 | 0.9392 | 0.5639 | n/a | n/a | n/a | 0 |
| small | blur | 5 | zero_dce_trained | 654 | 0.3518 | 1.0437 | 0.4961 | n/a | n/a | n/a | 0 |
| small | indoor_haze | 1 | clahe | 654 | 0.2037 | 0.7472 | 0.6967 | n/a | n/a | n/a | 0 |
| small | indoor_haze | 1 | none | 654 | 0.2061 | 0.7523 | 0.6920 | n/a | n/a | n/a | 0 |
| small | indoor_haze | 1 | zero_dce | 654 | 0.3497 | 1.0503 | 0.4829 | n/a | n/a | n/a | 0 |
| small | indoor_haze | 1 | zero_dce_trained | 654 | 0.3271 | 1.0100 | 0.5151 | n/a | n/a | n/a | 0 |
| small | indoor_haze | 3 | clahe | 654 | 0.2673 | 0.9026 | 0.5841 | n/a | n/a | n/a | 0 |
| small | indoor_haze | 3 | none | 654 | 0.2596 | 0.8911 | 0.5872 | n/a | n/a | n/a | 0 |
| small | indoor_haze | 3 | zero_dce | 654 | 0.3371 | 1.0129 | 0.5014 | n/a | n/a | n/a | 0 |
| small | indoor_haze | 3 | zero_dce_trained | 654 | 0.3302 | 1.0079 | 0.5032 | n/a | n/a | n/a | 0 |
| small | indoor_haze | 5 | clahe | 654 | 0.2904 | 0.9646 | 0.5343 | n/a | n/a | n/a | 0 |
| small | indoor_haze | 5 | none | 654 | 0.2858 | 0.9476 | 0.5397 | n/a | n/a | n/a | 0 |
| small | indoor_haze | 5 | zero_dce | 654 | 0.3290 | 0.9934 | 0.5131 | n/a | n/a | n/a | 0 |
| small | indoor_haze | 5 | zero_dce_trained | 654 | 0.3271 | 0.9974 | 0.5079 | n/a | n/a | n/a | 0 |
| small | low_light | 1 | clahe | 654 | 0.1243 | 0.5206 | 0.8455 | n/a | n/a | n/a | 0 |
| small | low_light | 1 | none | 654 | 0.1198 | 0.5056 | 0.8524 | n/a | n/a | n/a | 0 |
| small | low_light | 1 | zero_dce | 654 | 0.1249 | 0.5339 | 0.8440 | n/a | n/a | n/a | 0 |
| small | low_light | 1 | zero_dce_trained | 654 | 0.2164 | 0.7242 | 0.6828 | n/a | n/a | n/a | 0 |
| small | low_light | 3 | clahe | 654 | 0.2512 | 0.8176 | 0.6201 | n/a | n/a | n/a | 0 |
| small | low_light | 3 | none | 654 | 0.2570 | 0.8300 | 0.6162 | n/a | n/a | n/a | 0 |
| small | low_light | 3 | zero_dce | 654 | 0.2548 | 0.8295 | 0.6186 | n/a | n/a | n/a | 0 |
| small | low_light | 3 | zero_dce_trained | 654 | 0.3071 | 0.9422 | 0.5444 | n/a | n/a | n/a | 0 |
| small | low_light | 5 | clahe | 654 | 0.3289 | 0.9973 | 0.5100 | n/a | n/a | n/a | 0 |
| small | low_light | 5 | none | 654 | 0.3469 | 1.0481 | 0.4709 | n/a | n/a | n/a | 0 |
| small | low_light | 5 | zero_dce | 654 | 0.3501 | 1.0579 | 0.4639 | n/a | n/a | n/a | 0 |
| small | low_light | 5 | zero_dce_trained | 654 | 0.3132 | 0.9601 | 0.5338 | n/a | n/a | n/a | 0 |
| small | sensor_noise | 1 | clahe | 654 | 0.1119 | 0.4773 | 0.8706 | n/a | n/a | n/a | 0 |
| small | sensor_noise | 1 | none | 654 | 0.1069 | 0.4633 | 0.8798 | n/a | n/a | n/a | 0 |
| small | sensor_noise | 1 | zero_dce | 654 | 0.1112 | 0.4742 | 0.8703 | n/a | n/a | n/a | 0 |
| small | sensor_noise | 1 | zero_dce_trained | 654 | 0.1854 | 0.6428 | 0.7382 | n/a | n/a | n/a | 0 |
| small | sensor_noise | 3 | clahe | 654 | 0.1221 | 0.5015 | 0.8531 | n/a | n/a | n/a | 0 |
| small | sensor_noise | 3 | none | 654 | 0.1144 | 0.4818 | 0.8675 | n/a | n/a | n/a | 0 |
| small | sensor_noise | 3 | zero_dce | 654 | 0.1171 | 0.4864 | 0.8616 | n/a | n/a | n/a | 0 |
| small | sensor_noise | 3 | zero_dce_trained | 654 | 0.1774 | 0.6228 | 0.7498 | n/a | n/a | n/a | 0 |
| small | sensor_noise | 5 | clahe | 654 | 0.1415 | 0.5461 | 0.8109 | n/a | n/a | n/a | 0 |
| small | sensor_noise | 5 | none | 654 | 0.1230 | 0.5041 | 0.8483 | n/a | n/a | n/a | 0 |
| small | sensor_noise | 5 | zero_dce | 654 | 0.1282 | 0.5125 | 0.8376 | n/a | n/a | n/a | 0 |
| small | sensor_noise | 5 | zero_dce_trained | 654 | 0.1859 | 0.6494 | 0.7312 | n/a | n/a | n/a | 0 |
