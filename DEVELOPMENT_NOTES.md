# Development Notes

Some notes on the development process and findings.

## Debugging Journey

Started with a 128x150 grid which was way too small - the physics never reached the scaling regime properly. Bumped it up to 512x500 and things started working.

Also had an issue where I was rejecting samples if the measured scaling exponents didn't match theoretical values. Turns out that's dumb for finite-size systems - they never match theory exactly. Removed that validation and went from 58% valid samples to 100%.

## What Actually Works

After fixing the grid size and validation issues:
- All models hit 100% accuracy on the 2-class problem (EW vs KPZ)
- The top features aren't the scaling exponents (alpha, beta) like you'd expect
- Instead it's morphological stuff: gradient_variance, width_change, std_height
- Makes sense in hindsight - these are more robust for finite systems

## Robustness Testing

Ran some tests to see where it breaks:

**System size sweep** (L = 32 to 512):
- Still gets 98%+ accuracy even at L=32
- Meanwhile scaling exponent errors are 45-90% at small sizes
- So ML beats traditional analysis for small systems

**Noise variation** (0.1 to 5.0):
- 100% across the board, doesn't care about noise amplitude

**Crossover regime** (lambda = 0 to 1):
- Tested the EW to KPZ transition
- Stays above 99% through the whole crossover

## Lessons Learned

1. Run the actual pipeline before claiming anything works
2. Finite-size physics is different from asymptotic theory
3. Feature importance can be surprising - don't assume exponents are best

## Known Limitations

- Only 2 classes (removed Ballistic Deposition since it's same universality class as KPZ)
- No hyperparameter tuning, just sklearn defaults
- Could use more samples for a real publication
