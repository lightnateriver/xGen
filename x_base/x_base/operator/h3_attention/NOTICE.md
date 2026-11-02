# Optional H3 attention sources

`veda_vendor.py` derives from `src/h3_ascend/_vendor/veda/backend_v2.py` at
Ascend-H3 revision `e02a57325029e67eafaf67d64b3d62a330f1e8d4`, distributed
under Apache-2.0. The changes inject one shared immutable predictor bundle
and qualify helper imports. Operator mathematics and learned routing are retained.

The optional external `h3_ascend` source checkout must use that revision.
Its third-party notices and individual CANN/Miowtion/Spark licenses still apply.
The later noncommercial project license is not used for this import.
No model weights or proprietary deployment configuration are redistributed.
