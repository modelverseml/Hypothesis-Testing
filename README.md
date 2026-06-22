# ML Hypothesis Testing

Have you ever come across p-values, t-statistics, or F-values while building regression or logistic models and wondered what they actually mean? Do they impact the model’s performance? Why do people often suggest removing features with a p-value greater than 5%?

All of these questions tie back to hypothesis testing. In this discussion, we’ll break down these concepts step by step and explore them in depth.

## Repository Structure

```
Hypothesis-Testing/
├── README.md
├── requirements.txt
├── main.py                        # runs all examples + the power plot
├── hypothesis_tests/              # the package (importable)
│   ├── __init__.py
│   ├── z_test.py                  # one/two-sample z-tests
│   ├── t_test.py                  # one-sample, two-sample, paired t-tests
│   ├── chi_square_test.py         # goodness-of-fit and independence tests
│   ├── anova_f_test.py            # one-way ANOVA + F-test for variances
│   └── effect_size_power.py       # Cohen's d, Cramer's V, power analysis
├── examples/
│   ├── worked_examples.py         # runnable example for every section below
│   └── power_curves.py            # plots power vs. sample size
└── images/
```

## Getting Started

```bash
git clone https://github.com/modelverseml/Hypothesis-Testing.git
cd Hypothesis-Testing
pip install -r requirements.txt

# Run everything: all worked examples + the power-curves plot
python main.py

# Or run the pieces individually
python examples/worked_examples.py        # every worked example
python examples/power_curves.py           # power curves -> images/power_curves.png
```

### Using the package in your own code

```python
from hypothesis_tests import TTest

t_test = TTest(alpha=0.05, tail="two")
result = t_test.one_sample([32, 31, 29, 33, 30], pop_mean=30)
print(result["p_value"], result["reject_null"])
```

Each test returns a plain dictionary (statistic, degrees of freedom, p-value,
critical value, and the reject/fail-to-reject decision).

## What is Hypothesis Testing?

In statistics and machine learning, we often estimate population parameters (like means, variances, or regression coefficients) using sample data. But how do we know whether these estimates truly reflect the population? Can we trust them?

That’s where hypothesis testing comes in—it provides a structured way to test assumptions, validate results, and decide whether our findings are statistically significant or just due to random chance.


## Steps in Hypothesis Testing  

The basic steps in hypothesis testing begin with two statements:  

- **H<sub>0</sub> (Null Hypothesis):** The starting assumption  
  *(always written with =, ≤, or ≥)*  

- **H<sub>1</sub> (Alternative Hypothesis):** The opposite of the initial assumption  
  *(always written with ≠, >, or <)*  


### Example  

Let’s say Amazon sales in **July 2025** were claimed to be **is $10 billion**.  

- **H<sub>0</sub> :** Total sales = $10 billion  
- **H<sub>1</sub> :** Total sales ≠ $10 billion  

Here, we have set up a hypothesis about our claim.

When testing hypotheses, there are only two possible outcomes:  

- **Reject the Null Hypothesis (H<sub>0</sub>)**  
- **Fail to Reject the Null Hypothesis (H<sub>0</sub>)**  

**Important**: There is **no such thing as “accepting” the null hypothesis**.  
We can only say that we **do not have enough evidence to reject it**.  

--- 
### Critical Values and Test Types  

Let’s say our **sample valuation** comes out to be **$9 billion**.  
Since it’s based on a sample, there’s always some error. Assume the error margin is **± $1 billion**.  

That means our possible **population range** could be between **$7 billion and $10 billion**.  

### Critical Values  

When we test a claim, we define **critical values**:  

- **LCV (Lower Critical Value)**  
- **UCV (Upper Critical Value)**  

These values act as **boundaries** that help us decide whether to reject the null hypothesis.  


### Two-Tailed Test (≠ case)  

If our claim is an **equation or equality** (e.g., *H<sub>0</sub>: Sales = $10B*),  
we are interested in **both directions** (greater or smaller than the claim).  

- **Reject H<sub>0</sub>** if the observed value lies **below LCV** or **above UCV**.  

### One-Tailed Test (≤ or ≥ case)  

If our claim is **one-sided**, then we only care about **one direction**.  

- **Left-tailed test (≤)** → Reject H<sub>0</sub> if observed value < LCV.  
- **Right-tailed test (≥)** → Reject H<sub>0</sub> if observed value > UCV.
  
<p align="center">
<img src="images/critical_values.webp" alt="Critical Value" width="50%"/>
</p>

---

### Making Decisions:

There are different methods to make a decision for the given claims.

#### a) Critical Value Method:

 - First, we need to assume an error margin, which we call the significance level (α).
 - Let’s take α = 5%. Based on this α and the type of test, we calculate the Lower Critical Value (LCV) and Upper Critical Value (UCV).
 - Finally, we make a decision by comparing the sample mean (x̄) with the calculated critical values.

<p align="center">
<img src="images/critical_value_method.png" alt="Critical Values" height = "350" width="45%"/>
<img src="images/z-table.png" alt="Standard Deviation Values" height="350" width="45%"/>
</p>

Example : 
- **H<sub>0</sub> :** Total sales = $10 billion
- **H<sub>1</sub> :** Total sales ≠ $10 billion
- α=0.05 , x̄ = 9, σ=2, n=40

Formula

  LCV = μ<sub>0</sub>​ − z<sub>α/2</sub>​⋅(σ / √n)

  ​UCV = μ<sub>0</sub>​ + z<sub>α/2</sub>​⋅(σ / √n)

  We need the z-score that cuts off the upper 2.5% of the distribution.

  - The cumulative probability at that point is 1 − 0.025 = 0.975

  - Looking up in the z-table: 𝑧 = 1.96

    Steps
     - Standard Error (SE) = (σ / √n) = 2 /√40 ≈0.316
     - Critical values
       - Margin of error = z<sub>α/2</sub>​⋅(σ / √n) = 1.96 × 0.316 ≈ 0.619
       - 𝐿𝐶𝑉 = 10 − 0.619 = 9.381
       - UCV=10+0.619=10.619

So, the acceptance region is: [9.381, 10.619]

   - Compare sample mean
     - Since  9<9.381, it falls outside the acceptance region.
     - Decision: Reject H<sub>0</sub>
  

---
 #### b) P-Value Method:

The p-value is the probability of obtaining a test statistic at least as extreme as the one observed, assuming the null hypothesis is true.

- A high p-value indicates strong evidence to fail to reject H<sub>0</sub>.

- A low p-value (less than the chosen significance level α) provides evidence to reject H<sub>0</sub>.

Steps
- Calculate the Z-score for the sample mean.
- Find the p-value from the cumulative probability of the given Z-score using the Z-table.
- For a two-tailed test, multiply the p-value by 2.
- Decision rule: If  p≤α, reject H<sub>0</sub>; otherwise, fail to reject H<sub>0</sub>.
  
<p align="center">
<img src="images/p_value_method.png" alt="P-Value Method" height="350" width="45%"/>
 <img src="images/z-table.png" alt="Standard Deviation Values" height="350" width="45%"/>
</p>


Example : 
- **H<sub>0</sub> :** Total sales ≤ $10 billion
- **H<sub>1</sub> :** Total sales > $10 billion
- α=0.05 , x̄ = 9, σ=2, n=40

Formula

  Z-Score  = (x̄ -  μ<sub>0</sub>​) / (σ / √n)
  
  Steps
  
   - Standard Error (SE) = (σ / √n) = 2 /√40 ≈0.316
   - Z -Score : (9 - 10) / (0.316) ≈−3.16
   - Find cumulative probability (From the Z-table):
     - P(Z<−3.16)≈0.0008
   - Since it’s a two-tailed test:
     - p-value=2×0.0008=0.0016
   - Compare with α
     - p=0.0016<0.05
   - Decision: Reject H<sub>0</sub>
   ​
---
 #### c) T - Distribution:
 
It is also referred to as the Student’s t-distribution.

The t-distribution is generally used when the population standard deviation is unknown, and we estimate it using the sample standard deviation. It is especially useful for small sample sizes, as it accounts for the extra uncertainty introduced by estimating σ

- Use Z-test / P - Value : when population σ is known.

- Use T-test: when population σ is unknown, and you estimate it with the sample standard deviation s

Formula
t = (x̄ -  μ<sub>0</sub>​) / (S / √n)

Steps:
- Calculate the test statistic (t-score)
- Find degrees of freedom (df) = (n-1)
- Determine the critical value(s) from the t-table (based on α, one-tailed or two-tailed, and df)
- Decision rule

 - If ∣t∣>t <sub>critical</sub>, reject H <sub>0</sub>.
 - Otherwise, fail to reject H <sub>0</sub>.

<p align="center">
  <img src="images/t-distribution.png" alt="T - Distribution" height="350" width="45%"/>
  <img src="images/T-table.png" alt="T - Table" height="350" width="45%"/>
 
</p>

Example:

A company claims the average delivery time is 30 minutes. You take a random sample of 25 deliveries, and the results show:

- Sample mean (x̄) = 32 minutes

- Sample standard deviation (s) = 4 minutes

- Significance level (α) = 0.05

Steps : 
H<sub>0</sub>​:μ=30
H<sub>1</sub>​:μ=30
- Test statistic (t-score)
  - t =   (x̄ -  μ<sub>0</sub>​) / (S / √n) = (32−30)/(4 / √25) = 2.5
- df : n-1 = 24
- Critical value from t-table
  - At α=0.05 (two-tailed) and df=24:
     - t<sub>0.025,24</sub>​≈2.064
- Decision : Since t=2.5>2.064, we reject H<sub>0</sub>

---
d) Chi-Square Test (Goodness-of-Fit / Categorical counts) :

compare observed counts in categories to expected counts , if observed counts are much different from expected, reject the assumed distribution.

Formula

χ<sup>2</sup> = ∑( (O−E)<sup>2</sup>/E​ )
- E → Expected
- O → Observed

<p align="center">
<img src="images/Chi-square.webp" alt="Chi Square Distribution" height="350" width="45%"/>
 <img src="images/Chi-square_distribution_table.webp" alt="Chi Squared Values" height="350" width="45%"/>
</p>


Example : 
claim : We want to test if a coin is fair (i.e., probability of heads = 0.5)

Number of times coin flipped = 100

Steps : 

H<sub>0</sub> : p = 0.5
H<sub>1</sub> : p ≠ 0.5

Let say after fliping 100 times observed is 60 heads , 40 tails

 - Observed Heads (O<sub>1</sub>) = 60

 - Observed Tails (O<sub>2</sub>) = 40

Expected :
 - Expected Heads (E<sub>1</sub>) = 50
 - Expected Heads (E<sub>2</sub>) = 50


Steps : 
- χ<sup>2</sup> = ∑( (O−E)<sup>2</sup>/E​ ) = (60 - 50)<sup>2</sup> / 50 + (40 - 50)<sup>2</sup> / 50 = 4
- Degrees of Freedom : k(Number of categories)−1=2−1=1
- For α = 0.05 and df = 1:
   - Critical value (χ<sup>2</sup><sub>0.05</sub>,1) ≈ 3.841

- Since χ² (4) > 3.841 → reject H<sub>0</sub> .

---

### e) Two-Sample Tests

In ML we often want to compare two distributions: scores from model A vs. model B, response times before vs. after a change, accuracy of treatment vs. control. These are two-sample tests.

#### Independent two-sample t-test

Used when the two samples are drawn independently (e.g., two different sets of users).

Formula (equal variances, pooled):

t = (x̄<sub>1</sub> − x̄<sub>2</sub>) / √( s<sup>2</sup><sub>p</sub> · (1/n<sub>1</sub> + 1/n<sub>2</sub>) )

where s<sup>2</sup><sub>p</sub> = ((n<sub>1</sub>−1)·s<sub>1</sub><sup>2</sup> + (n<sub>2</sub>−1)·s<sub>2</sub><sup>2</sup>) / (n<sub>1</sub>+n<sub>2</sub>−2)

- df = n<sub>1</sub> + n<sub>2</sub> − 2
- If variances are unequal, use **Welch's t-test** (Satterthwaite df, no pooling).

Example: Model A average accuracy 0.82, Model B average accuracy 0.85 over 30 runs each. Does the difference reflect a real improvement or noise? See `hypothesis_tests/t_test.py → TTest.two_sample_independent`.

#### Paired t-test

Used when the same units are measured twice (before/after a training, A/B on the same users).

Formula: apply a one-sample t-test on the within-pair differences d<sub>i</sub> = x<sub>i,after</sub> − x<sub>i,before</sub> against μ<sub>0</sub> = 0.

t = d̄ / (s<sub>d</sub> / √n)  ,  df = n − 1

Pairing removes between-subject variance, so paired tests usually have more statistical power than independent ones when applicable.

---

### f) ANOVA (Analysis of Variance) / F-Test

When we have **three or more groups**, running pairwise t-tests inflates the false-positive rate. ANOVA tests whether *any* group mean differs:

- **H<sub>0</sub>:** μ<sub>1</sub> = μ<sub>2</sub> = … = μ<sub>k</sub>
- **H<sub>1</sub>:** at least one μ<sub>i</sub> differs

The F-statistic compares variance **between** groups to variance **within** groups:

F = MS<sub>between</sub> / MS<sub>within</sub>

where
- SS<sub>between</sub> = Σ n<sub>i</sub> · (x̄<sub>i</sub> − x̄)<sup>2</sup>
- SS<sub>within</sub>  = Σ Σ (x<sub>ij</sub> − x̄<sub>i</sub>)<sup>2</sup>
- df<sub>between</sub> = k − 1
- df<sub>within</sub>  = N − k
- MS = SS / df

<p align="center">
<img src="images/F-distribution.png" alt="F Distribution" height="400" width="45%"/>
<img src="images/F-table.png" alt="F Table" height="400" width="45%"/>
</p>

**Decision rule:** if F > F<sub>critical</sub> (or p < α), reject H<sub>0</sub>.

Example: comparing three training algorithms over 25 evaluation runs each, see `hypothesis_tests/anova_f_test.py → AnovaFTest.one_way`.

The same F-distribution underlies the **F-test for equality of variances** (ratio of two sample variances), useful as a pre-check before deciding between Student's and Welch's t-test.

---

### g) Effect Size & Statistical Power

A p-value tells you **whether** a difference is unlikely under H<sub>0</sub>; it does **not** tell you **how large** the difference is. With a large enough sample, even trivial differences become "significant." That is why every hypothesis test should be reported alongside an **effect size**.

#### Common effect sizes

| Test | Effect size | Formula | Small / Medium / Large |
|---|---|---|---|
| One-sample t | Cohen's d | (x̄ − μ<sub>0</sub>) / s | 0.2 / 0.5 / 0.8 |
| Two-sample t | Cohen's d | (x̄<sub>1</sub> − x̄<sub>2</sub>) / s<sub>pooled</sub> | 0.2 / 0.5 / 0.8 |
| Two proportions | Cohen's h | 2·arcsin(√p<sub>1</sub>) − 2·arcsin(√p<sub>2</sub>) | 0.2 / 0.5 / 0.8 |
| Chi-square independence | Cramer's V | √(χ<sup>2</sup> / (n · min(r−1, c−1))) | 0.1 / 0.3 / 0.5 |

#### Statistical Power

Power = P(reject H<sub>0</sub> | H<sub>1</sub> is true) = 1 − β

It depends on four interlinked quantities — fix any three and the fourth is determined:

1. **α** (significance level)
2. **Effect size**
3. **Sample size (n)**
4. **Power (1 − β)** — typically 0.80 by convention

Common uses:
- **Sample size planning**: given a target effect size and α=0.05, power=0.80, how many observations do I need?
- **Post-hoc check**: my experiment failed to reject H<sub>0</sub> — was it underpowered, or is there genuinely no effect?
- **Minimum detectable effect**: with my fixed n, what is the smallest effect I can reliably catch?

See `hypothesis_tests/effect_size_power.py → PowerAnalysis` and `examples/power_curves.py` for the power curves plot.

---

## Choosing the Right Test

| Scenario | Test |
|---|---|
| One sample, σ known | Z-test |
| One sample, σ unknown | One-sample t-test |
| Two independent samples | Independent t-test (pooled or Welch) |
| Paired observations | Paired t-test |
| 3+ group means | One-way ANOVA |
| Equality of two variances | F-test for variances |
| Categorical distribution vs. expected | Chi-square goodness-of-fit |
| Two categorical variables independent? | Chi-square independence |

---

## References

- Casella, G. & Berger, R. L. — *Statistical Inference*
- Cohen, J. — *Statistical Power Analysis for the Behavioral Sciences*
- SciPy documentation: `scipy.stats`
- statsmodels documentation: `statsmodels.stats.power`
