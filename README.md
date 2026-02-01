# sandbox
Contains simpler code used to benchmark miscellaneous concepts.

## Spectral Bloom Filter
Path: [`spectral-bloom-filter`](https://github.com/eventuallyconsistentwrites/sandbox/tree/main/spectral-bloom-filter)

Executing
1. Testing the filter with increasing filter size against random and zipf distributions

`python3 -m spectral-bloom-filter.Main`

2. Testing the filter with constant filter size but increasing number of elements inserted from random and zipf distributions
   
`python3 -m spectral-bloom-filter.MainV2`

## Count-Min Sketch
Path: [`count-min-sketch`](https://github.com/eventuallyconsistentwrites/sandbox/tree/main/count-min-sketch)

Executing
1. Testing the sketch with increasing size against random and zipf distributions

`python3 -m count-min-sketch.Main`

2. Testing the sketch with constant size but increasing number of elements inserted from random and zipf distributions

`python3 -m count-min-sketch.MainV2`

## Common
Path: [`common`](https://github.com/eventuallyconsistentwrites/sandbox/tree/main/common)

Contains helper classes and functions used by other classes.