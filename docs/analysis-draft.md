
## Initial analysis on the string matching

Implementation of the following algorithms was done
1. Brute force
2. KMP

And following optimizations were considered
1. Early breaking in brute force
2. Pivoted layout in both brute force and KMP


## Bottleneck investigation

For the simple brute force implementation, the memory pressure was mostly on the L1 cache, due to this the pivoted layout was not of much use although the loads we coalesced, the cost of calculating the offsets dominated the compute time and performed worse than the non-pivoted layout, increasing the number of instructions. 
Things to try:
- A simpler pivoted layout
- Load 4 characters at once, to reduce number of loads by 1/4. 

