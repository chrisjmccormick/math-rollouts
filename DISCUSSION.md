### Misc Updates

- README should note that "math rollouts" refers to the standard Math12k dataset and the Math500 benchmark.
- Broad purposes:
   - Analyzing impact of RL training, 
   - Analyzing model behavior on reasoning tasks.
   - Baseline comparisons for RL experiments.

### Topics Explored

**Difficulty Banding**

The Math12k dataset defines problem difficulty levels L1-L5, but we've re-classified problems based on model performance. 

This is valuable information for RL training:
- For training samples, it clarifies whether there is room for improvement.
- For test samples, it clarifies where we're seeing gains, and whether we've regressed on easier problems.

**Evaluating Openers**

Some of the rollouts generated here are done by uniformly sampling from a token nucleus rather than using the model's probabilities.

In particular, we've measured the accuracy of different opening branches to see whether they differ. They matter a surprising amount! This is a phenomenon that's been observed in the literature.

- [PPPO](https://gcatnjust.github.io/ChenGong/paper/sun_aaai26.pdf) is an algorithm designed specifically to identify and reinforce the best openers.
- [Best-of-N](https://arxiv.org/abs/2505.14216) is a technique that can be used to improve the quality of the opener.

**Reachability**

We've reconstructed the experiment performed in _[The limits of RLVR](https://proceedings.neurips.cc/paper_files/paper/2025/file/537d5aa768c2d534016a4d06f87bc8fb-Paper-Conference.pdf)_ showing that Qwen 2.5 Math 1.5B outperforms the RL-tuned Oat-Zero model when evaluated at Pass@256.

Oat-Zero has a much higher pass@k for lower values of k, but sacrifices access to solutions to the most difficult problems.

Future references:
- Also covered in [Rl vs. Distillation](https://arxiv.org/abs/2505.14216).
    - RLVR causes "branch collapse", producing more consistently accuracte results for easier problems, but (failing to explore?). 

**New Branches and Closed Branches**

- Usually handled via entropy analysis.
- Simpler approach--under standard sampling, does every token in the sequence exist within the model's sampling nucleus at that position?
    - If the base model has the token but the tuned model does not, then the tuned model has **closed** that branch. 
    - If the tuned model's nucleus contains a token not present in the base model's nucleus, it has **opened** a new branch.
- We apply this analysis by applying the math model and oat-zero model to eachothers correct and incorrect responses.

### Data Coverage

_Natural Sampling_

Qwen2.5-Math-1.5B:
- All of math12k (L4-L5) at K=?
- Math500 at K=?
    - Additionally, hardest problems extended to K=256 to perform pass@k

"sail/Qwen2.5-Math-1.5B-Oat-Zero" Oat-Zero:
- TODO - math12k (L4-L5) 
- Math500 at K=?
    - TODO - pass at k?

Distill R1:

Qwen3-8B-Base:

Qwen3-8B:
