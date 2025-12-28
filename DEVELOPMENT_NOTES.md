# Dev Notes

Quick writeup of what I learned building this.

## The bugs I had to fix

The original grid was 128x150 which turned out to be way too small. The surfaces never actually got into the proper scaling regime, so all the physics measurements were garbage. Increased it to 512x500 and suddenly things made sense.

Bigger problem though - I had validation code that was rejecting samples where α or β didn't match the theoretical predictions. This seems reasonable until you remember that *no finite system ever matches asymptotic theory*. That's literally the point. Was throwing away 42% of my data because of this. Removed the bounds check and everything works now.

## Interesting findings

The features that actually matter for classification aren't what I expected. I thought the scaling exponents (α, β) would be the most important since that's what the physics literature focuses on. Nope. 

Top 3 features by importance:
1. gradient_variance (18%)
2. width_change (19%)  
3. std_height (13%)

The exponents barely register. Makes sense though - they're noisy as hell in finite systems. The morphological features are way more stable.

## Testing how robust this is

Wanted to see where the classification breaks down:

Tried different system sizes from L=32 up to 512. Even at L=32 it gets 98% accuracy, while the scaling exponent fits have like 45% error. Pretty clear that ML is doing something smarter than just measuring exponents.

Varied noise amplitude by 50x (0.1 to 5.0) and it didn't care at all. 100% across the range.

Also tested the crossover between EW and KPZ by gradually turning on the nonlinearity. Classification stays solid through the whole transition, even catches the intermediate regime at ~99.6%.

## What I'd change

Only using 2 universality classes right now. Originally had 3 but realized Ballistic Deposition is the same class as KPZ (oops). Could add more but these two are enough to prove the concept.

Haven't done any hyperparameter tuning, just using sklearn defaults. They work fine so not sure it's worth it.

Sample size is decent for a demo but would need more for a proper paper.
