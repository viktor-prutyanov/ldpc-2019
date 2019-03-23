# LDPC 2019

### Reliability-based decoding algorithms of non-binary LDPC codes

Low-density parity-check (LDPC) codes over GF(q) have advantages over binary LDPC codes. Daveyand MacKay were first who used belief propagation (BP) to decode such codes. They showed that non-binary LDPC codes significantly outperform their binary counterparts. Moreover, non-binary LDPC codesare especially good for the channels with burst errors and high-order modulations.  Unfortunately, theirdecoding complexity is still large, that is why iterative hard and soft-reliability based decoding majorityalgorithms are of considerable interest for high-throughput practical applications.

To run simulation and plot results:

```
    ./make_noise --num 500 --num-points 50
    parallel -k < cmds.txt
    ./plot.py
```
