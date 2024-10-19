## Phase 1 

1. Implement Brute force string search in GPU.
    - Experiment with data layouts and see the performance.
2. Implement KMP for string matching in GPU

Primary source for reference: https://dl.acm.org/doi/10.1007/s00778-015-0409-y

## Phase 2

1. Try out FFT for string matching.


## October 19

Things done till now
- Brute force 
    - Without early breaking (saturates compute and memory)
    - With early breaking
    - Pivoted layout
- KMP
    - KMP basic
    - KMP step
    - Pivoted layout

Things yet to try
- Discover possibility of using shared memory either for the pattern or the main strings.
- Increase the pivot size to see if there are reductions in load instructions



