# Optimization benchmark

| model                      |   accuracy |   f1_macro |   p95_latency_ms |   size_mb |
|:---------------------------|-----------:|-----------:|-----------------:|----------:|
| original (fp32, torch)     |       0.48 |   0.470814 |             73   |     541   |
| distilled (6-layer, torch) |       0.43 |   0.241078 |             41.4 |     372.5 |
| INT8 (ONNX Runtime)        |       0.3  |   0.266482 |             73.3 |     136   |
