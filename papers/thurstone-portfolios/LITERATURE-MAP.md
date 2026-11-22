# Literature map: Portfolio-Implied Contests

The paper *Portfolio-Implied Contests, with an Application to Tail Risk* reads a long-only
portfolio as the winning probabilities of a race, and uses that reading to adjust a benchmark
for a view about dependence. The adjustment is Thurstone portfolio polishing, which the paper
calls the dependence tilt.

This file lists the work the paper draws on and the work it differs from. The interactive
version is at <https://allocation.microprediction.org/maps/portfolio-implied-contests.html>.
Its data are the `NODES` and `EDGES` arrays in `docs/maps/portfolio-implied-contests.html`;
keep the two in step.

## This paper and its inverse

- **[Portfolio-Implied Contests](https://allocation.microprediction.org/papers/thurstone-portfolios/paper.pdf)** (2026). Cotton 2026, 'Portfolio-Implied Contests, with an Application to Tail Risk' (SSRN 7737202). A long-only portfolio is read as the winning probabilities of a race among its assets, and the abilities behind it are recovered at a cost linear in the number of assets.
- **[Thurstone portfolio polishing](https://allocation.microprediction.org/methods/thurstone-portfolio-polishing/index.html)** (2026). The paper's application, which it calls the dependence tilt. Calibrate a race to a benchmark under a reference law, hold the abilities, replace the dependence with a target law that may carry tail dependence, and race again. No covariance is inverted and no program is re-solved.
- **[Cotton 2026](https://arxiv.org/abs/2609.01133)** (2026). 'Scalable Inversion of Contests with Correlated Performances' (arXiv:2609.01133). One survival field shared by all competitors gives every winning probability in one pass. Factor, group, nested and tree dependence are handled by conditioning. The inverse is an iteration on this forward map.
- **[Cotton 2021](https://doi.org/10.1137/19M1276261)** (2021). 'Inferring Relative Ability from Winning Probability in Multientrant Contests', SIAM J. Financial Math. 12(1):295-317. Winning probabilities to abilities on a lattice with exact tie handling, for independent performances.

## Reverse and inverse optimization

- **[Sharpe 1974](https://scholar.google.com/scholar?q=Sharpe+1974+Imputing+Expected+Security+Returns+from+Portfolio+Composition)** (1974). JFQA 9(3):463-472. Imputes the expected returns that make held weights mean-variance optimal. The covariance is given.
- **[Black–Litterman 1992](https://doi.org/10.2469/faj.v48.n5.28)** (1992). FAJ 48(5):28-43. Implied returns from the market portfolio, then a Bayesian update on views. Both the reverse step and the views are on returns; the covariance is an input.
  - To Thurstone portfolio polishing (same object, two readings): anchor to a benchmark: view on returns ↔ view on dependence.
- **[Idzorek 2005](https://scholar.google.com/scholar?q=Idzorek+step-by-step+guide+Black-Litterman+model)** (2005). Practitioner guide to Black-Litterman, with a method for setting view confidence. Views are on returns.
- **[Qian–Gorman 2001](https://doi.org/10.2469/faj.v57.n2.2432)** (2001). FAJ 57(2):44-51. Black-Litterman with views on volatilities and correlations, giving a conditional covariance. The view is on second moments and the weights still come from inverting a covariance.
  - To Thurstone portfolio polishing (special case): a correlation view is the second-moment case.
- **[Ahuja–Orlin 2001](https://doi.org/10.1287/opre.49.5.771.10607)** (2001). Oper. Res. 49(5):771-783. Inverse optimization: recover the objective that makes an observed solution optimal.
- **[Bertsimas et al 2012](https://doi.org/10.1287/opre.1120.1115)** (2012). Oper. Res. 60(6):1389-1403. Recasts Black-Litterman as inverse optimization and extends it to coherent risk measures. The inverse is a conic program on returns and risk.
  - To Thurstone portfolio polishing (a step the cited work does not take): inverts for returns and risk, not for dependence.
- **[Yu–Wang–Dong 2023](https://doi.org/10.1016/j.ribaf.2023.101879)** (2023). Res. Int. Bus. Finance 64:101879. Learns an investor's risk preference from holdings by inverting the mean-variance optimizer.
  - To Thurstone portfolio polishing (a different approach to the same problem): inverts the optimizer for a risk preference; polishing inverts a race for abilities.

## Choice models and share inversion

- **[Bradley–Terry 1952](https://doi.org/10.1093/biomet/39.3-4.324)** (1952). Biometrika 39:324-345. Paired-comparison model in which the odds of one item beating another are the ratio of their strengths.
- **[Luce 1959](https://scholar.google.com/scholar?q=Luce+1959+Individual+Choice+Behavior+A+Theoretical+Analysis)** (1959). Luce's choice axiom, independence of irrelevant alternatives. Capitalization weighting obeys it: the ratio of two weights does not depend on the rest of the universe.
  - To Portfolio-Implied Contests (a different approach to the same problem): capitalization weighting obeys Luce's axiom; a correlated race does not.
- **[Plackett 1975](https://doi.org/10.2307/2346567)** (1975). Applied Statistics 24(2):193-202. Extends Luce's model from choices to full rankings.
- **[Yellott 1977](https://doi.org/10.1016/0022-2496(77)90026-8)** (1977). J. Math. Psych. 15(2):109-144. Luce's model is equivalent to a race with independent Gumbel performances.
- **[McFadden 1974](https://scholar.google.com/scholar?q=McFadden+1974+conditional+logit+analysis+of+qualitative+choice+behavior)** (1974). Conditional logit. The inverse from shares to utilities is closed form, log share differences, under independence.
  - To Cotton 2021 (special case): log shares: the closed-form inverse for independent Gumbel noise.
- **[Berry 1994](https://doi.org/10.2307/2555829)** (1994). RAND 25(2):242-262. Logit and nested-logit share systems invert in closed form to mean utilities.
- **[Berry et al 1995 (BLP)](https://doi.org/10.2307/2171802)** (1995). Econometrica 63(4):841-890. With random coefficients the share inverse has no closed form and is computed by a contraction mapping.
  - To Cotton 2026 (a different approach to the same problem): share inversion by contraction on simulated shares.
- **[Berry et al 2013](https://doi.org/10.3982/ECTA10135)** (2013). Econometrica 81(5):2087-2111. 'Connected substitutes' conditions for a share system to be globally invertible.
  - To Cotton 2026 (builds on): conditions for a unique share inverse.

## Thurstonian race and probit computation

- **[Thurstone 1927](https://scholar.google.com/scholar?q=Thurstone+1927+A+law+of+comparative+judgment)** (1927). Psych. Rev. 34:273-286. The law of comparative judgment: each alternative has a latent ability and a noisy performance, and the extreme performance is chosen.
  - To Cotton 2021 (builds on): the race.
- **[Mosteller 1951](https://scholar.google.com/scholar?q=Mosteller+1951+Remarks+on+the+method+of+paired+comparisons+least+squares+equal+standard+deviations+equal+correlations)** (1951). Psychometrika 16:3-9. Least-squares solution of Thurstone's model with equal variances and equal correlations.
- **[Clark 1961](https://doi.org/10.1287/opre.9.2.145)** (1961). Oper. Res. 9(2):145-162. Moment-matching approximation to the maximum of correlated Gaussians.
  - To Cotton 2026 (a different approach to the same problem): moment-matching approximation of the maximum.
- **[Harville 1973](https://doi.org/10.1080/01621459.1973.10482425)** (1973). JASA 68:312-316. Assigns probabilities to the finishing orders of a multi-entry competition from the winning probabilities.
  - To Cotton 2021 (builds on): winning probabilities in multi-entry contests.
- **[Butler–Moffitt 1982](https://scholar.google.com/scholar?q=Butler+Moffitt+1982+computationally+efficient+quadrature+procedure+one-factor+multinomial+probit)** (1982). Econometrica 50(3):761-764. One-factor multinomial probit by quadrature over the factor.
  - To Cotton 2026 (builds on): condition on a factor and integrate by quadrature.
- **[GHK simulator](https://scholar.google.com/scholar?q=GHK+simulator+Geweke+Hajivassiliou+Keane+multivariate+normal+rectangle+probabilities)** (1994). Geweke (1989), Keane (1994), Hajivassiliou, McFadden and Ruud (1996). Simulates multivariate normal rectangle probabilities by sequential conditioning. The standard way to compute probit choice probabilities.
  - To Cotton 2026 (a different approach to the same problem): probit probabilities by simulation.
- **[Maydeu-Olivares 2005](https://scholar.google.com/scholar?q=Maydeu-Olivares+Bockenholt+2005+structural+equation+modeling+of+paired+comparison+and+ranking+data)** (2005). Psych. Methods 10(3):285-304. Estimates Thurstonian models of paired-comparison and ranking data by structural equation methods.

## Entropy and copula pooling

- **[Meucci 2006](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=848407)** (2006). Risk 19(9). Views on realizations of the market, with marginals joined by a copula. The posterior distribution is passed to an optimizer.
- **[Meucci 2008](https://arxiv.org/abs/1012.2848)** (2008). Risk 21(10). Views on means, correlations, CVaR, tail probabilities or rankings are constraints on scenario probabilities, and the posterior is the minimum relative entropy projection. A separate optimizer turns the posterior into weights.
  - To Thurstone portfolio polishing (a different approach to the same problem): tail views by re-weighting scenarios, then re-optimizing.

## Higher-moment, skew and copula Black–Litterman

- **[Giacometti et al 2007](https://scholar.google.com/scholar?q=Giacometti+Bertocchi+Rachev+Fabozzi+stable+distributions+Black-Litterman)** (2007). Quant. Finance 7(4):423-433. Black-Litterman with stable and Student-t priors and VaR or CVaR as the risk measure.
- **[Harvey et al 2010](https://doi.org/10.1080/14697681003756877)** (2010). Quant. Finance 10(5):469-485. Bayesian portfolio selection under a skew-normal likelihood, by MCMC and utility maximization.
- **[Park–Nguyen 2025](https://doi.org/10.1007/s11579-025-00387-1)** (2025). Math. Financial Econ. 19(2):407-444. The Black-Litterman posterior under hidden truncation is skew-normal in closed form, and is optimized with Simaan's three-moment risk model.
  - To Thurstone portfolio polishing (a step the cited work does not take): skew-normal posterior; no joint lower tail.
- **[Martellini–Ziemann 2010](https://doi.org/10.1093/rfs/hhp099)** (2010). RFS 23(4):1467-1502. Improved estimators of coskewness and cokurtosis for allocation with higher moments.
- **[Sahamkhadam et al 2022](https://doi.org/10.1016/j.ejor.2021.06.015)** (2022). EJOR 297(3):1055-1070. Black-Litterman with R-vine copulas, followed by mean-CVaR optimization. The copula supplies return scenarios; the reverse step is on returns.
  - To Thurstone portfolio polishing (a step the cited work does not take): copula scenarios, but the reverse step stays on returns.
- **[Kraus–Litzenberger 1976](https://doi.org/10.1111/j.1540-6261.1976.tb01961.x)** (1976). J. Finance 31(4):1085-1100. Three-moment CAPM: coskewness with the market is priced.
- **[Harvey–Siddique 2000](https://doi.org/10.1111/0022-1082.00247)** (2000). J. Finance 55(3):1263-1295. Conditional coskewness carries a premium in the cross-section.
- **[Simaan 1993](https://doi.org/10.1287/mnsc.39.5.568)** (1993). Mgmt Science 39(5):568-577. A three-parameter framework that admits skewness preference without the full comoment tensor.
- **[Lai 1991](https://doi.org/10.1007/BF02408382)** (1991). Rev. Quant. Finance Acc. 1(3):293-305. Polynomial goal programming for mean-variance-skewness portfolios.
- **[Briec et al 2007](https://doi.org/10.1287/mnsc.1060.0596)** (2007). Mgmt Science 53(1):135-149. A shortage function measures mean-variance-skewness efficiency.
  - To Thurstone portfolio polishing (a different approach to the same problem): skewness through comoment tensors and an optimizer.
- **[Boudt et al 2020](https://doi.org/10.1093/jjfinec/nby022)** (2020). J. Financial Econometrics 18(1):1-23. Shrinks the sample coskewness matrix toward structured targets.

## Tail-aware allocation

- **[Rockafellar–Uryasev 2000](https://doi.org/10.21314/JOR.2000.038)** (2000). J. Risk 2(3):21-41. Minimizes CVaR over scenarios by a linear program. Reaches the tail without inverting a covariance.
  - To Thurstone portfolio polishing (a different approach to the same problem): reaches the tail by a linear program.
- **[Raffinot 2018 (HERC)](https://scholar.google.com/scholar?q=Raffinot+2018+The+Hierarchical+Equal+Risk+Contribution+Portfolio)** (2018). Hierarchical equal risk contribution. Allocates down a cluster tree and admits downside risk measures.
  - To Thurstone portfolio polishing (a different approach to the same problem): reaches the tail by bisection on a tree.
- **[Lohre et al 2020](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3513399)** (2020). Hierarchical risk parity with lower-tail dependence as the linkage. Re-weights by recursive bisection with no covariance inversion. There is no view and no inverse map.
  - To Thurstone portfolio polishing (a different approach to the same problem): no covariance inversion, but no view and no inverse.
- **[Tail-dependence clustering 2018](https://doi.org/10.3390/risks6040115)** (2018). Risks 6(4):115. Clusters assets by lower-tail dependence, following De Luca and Zuccolotto (2011).
- **[Tail-risk parity](https://scholar.google.com/scholar?q=tail+risk+parity+expected+shortfall+contributions)** (2017). Equalizes expected-shortfall contributions across assets.
  - To Thurstone portfolio polishing (a different approach to the same problem): equalizes tail-risk contributions.

## Inversion-free and covariance-based allocators

- **[López de Prado 2016 (HRP)](https://doi.org/10.3905/jpm.2016.42.4.059)** (2016). J. Portfolio Mgmt 42(4):59-69. Hierarchical risk parity: clusters, orders and bisects, with no covariance inversion.
  - To Portfolio-Implied Contests (a different approach to the same problem): de-duplicates by clustering; comparator in the Dow walk-forward.
- **[Cotton 2024 (Schur)](https://arxiv.org/abs/2411.05807)** (2024). Schur-complementary allocation (arXiv:2411.05807). A one-parameter family between hierarchical risk parity and minimum variance.
  - To Portfolio-Implied Contests (a different approach to the same problem): comparator in the Dow walk-forward.
- **[Maillard et al 2010](https://doi.org/10.3905/jpm.2010.36.4.060)** (2010). J. Portfolio Mgmt 36(4):60-70. Equal risk contribution (risk parity).
  - To Portfolio-Implied Contests (a different approach to the same problem): comparator in the Dow walk-forward.
- **[Choueifaty–Coignard 2008](https://doi.org/10.3905/JPM.2008.35.1.40)** (2008). J. Portfolio Mgmt 35(1):40-51. Maximum diversification.
  - To Portfolio-Implied Contests (a different approach to the same problem): covariance-based comparator.

## Long-only allocation as regularized regression

- **[Britten-Jones 1999](https://doi.org/10.1111/0022-1082.00120)** (1999). J. Finance 54(2):655-671. Tangency weights are the coefficients of regressing a constant on excess returns.
- **[Jagannathan–Ma 2003](https://scholar.google.com/scholar?q=Jagannathan+Ma+2003+Risk+Reduction+in+Large+Portfolios+Why+Imposing+the+Wrong+Constraints+Helps)** (2003). J. Finance 58(4):1651-1683. The no-short constraint acts like shrinkage of the covariance.
- **[Brodie et al 2009](https://doi.org/10.1073/pnas.0904287106)** (2009). PNAS 106(30):12267-12272. An l1 penalty on the regression gives sparse, stable Markowitz portfolios.
  - To Portfolio-Implied Contests (a different approach to the same problem): l1 penalty ↔ a dependence-adjusted regularizer on the simplex.

## Perturbed optimizers

- **[Williams 1977](https://doi.org/10.1068/a090285)** (1977). Environment and Planning A 9(3):285-344. The expected maximum utility as a measure of user benefit in travel demand models.
- **[McFadden 1981](https://scholar.google.com/scholar?q=McFadden+1981+Econometric+models+of+probabilistic+choice)** (1981). The social-surplus function of a random-utility model: choice probabilities are the gradient of the expected maximum utility.
  - To Portfolio-Implied Contests (same object, two readings): weights are the gradient of an expected maximum.
- **[Berthet et al 2020](https://arxiv.org/abs/2002.08676)** (2020). NeurIPS 2020. Perturbed optimizers: a differentiable argmax obtained by adding noise to the inputs. Gumbel noise gives the entropy-regularized softmax.
  - To Portfolio-Implied Contests (special case): Gumbel noise gives softmax; correlated Gaussian noise gives the paper's regularizer.

## Empirical premise

- **[Longin–Solnik 2001](https://doi.org/10.1111/0022-1082.00340)** (2001). J. Finance 56(2):649-676. Rejects normality for the joint lower tail of international equity returns, but not for the upper tail.
  - To Thurstone portfolio polishing (builds on): premise: the joint lower tail is not Gaussian.
- **[Ang–Chen 2002](https://doi.org/10.1016/S0304-405X(02)00068-5)** (2002). J. Financial Econ. 63(3):443-494. Correlations between U.S. equity portfolios and the market are higher in downturns than a normal distribution implies.
  - To Thurstone portfolio polishing (builds on): premise: correlations rise in downturns.
- **[Embrechts et al 2002](https://scholar.google.com/scholar?q=Embrechts+McNeil+Straumann+2002+Correlation+and+dependence+in+risk+management+properties+and+pitfalls)** (2002). Sets out why correlation is an incomplete summary of dependence.
- **[Patton 2004](https://doi.org/10.1093/jjfinec/nbh006)** (2004). J. Financial Econometrics 2(1):130-168. Modelling asymmetric dependence gives economically significant gains in an asset-allocation problem.
  - To Thurstone portfolio polishing (builds on): premise: asymmetric dependence matters for allocation.
- **[Mallela–Leonelli 2026](https://arxiv.org/abs/2606.16840)** (2026). Hüsler-Reiss extremal graphs for equities: the lower-tail network is near-complete and the upper-tail network is thin.
  - To Thurstone portfolio polishing (builds on): premise: crash together, rally apart.

## Where the method sits

- Reverse optimization recovers expected returns from a portfolio, with the covariance given.
  Polishing recovers abilities, with the dependence given, and then changes the dependence.
- Entropy pooling and copula versions of Black–Litterman accept views about tails. They
  re-weight or simulate scenarios and then solve a program for the weights.
- Share inversion in the choice literature is closed form for the logit and iterative for
  correlated tastes. The inverse used here is deterministic for independent, factor, group,
  nested and tree dependence.
- Hierarchical and scenario methods reach the tail without inverting a covariance. They carry
  no benchmark anchor and no inverse map.

## Open questions in the paper

- The choice of reference law and of the strength of the view.
- Estimation error in the abilities, a counterpart of covariance shrinkage.
- A robust online estimate of downside covariance to drive the race.
- Whether a tail view improves drawdown or expected shortfall out of sample.

## Sources and checks

Entries come from the related-work section of the paper, the talk, and two earlier literature
sweeps. Links are DOIs or arXiv/SSRN pages where one was confirmed against Crossref, and
Google Scholar searches otherwise. Venues of entries not cited in the paper have not been
re-verified and should be checked before they are cited.
