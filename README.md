# Side Information Boosted Symbolic Regression

Side Information Boosted Symbolic Regression is a model enhanced from Deep Symbolic Optimization (DSO)[1, 2]. It incorporates side information into the Symbolic Regression task. Additionally, it includes a Side Information Generator, which can automatically generate side information if users are unsure about what side information to use.

### Getting Started

First, the DSO package needs to be installed. You can do so by following the instructions in the link below:

```
https://github.com/dso-org/deep-symbolic-optimization
```

After the DSO package is installed, necessary adaptations are required to the original DSO package to allow it to accept side information. The files for these adaptations are listed in the DSO-Adapted folder.

### Benchmark Problems

Datasets are generated using given benchmark expression functions. Python files in the benchmark folder provide examples on how to generate such datasets.

### Game Theory Problems

Our experiments focus on 2x2 and 3x3 bi-matrix games. We use Nashpy to generate Game Theory datasets. You can install it with:

```
python -m pip install nashpy
```

For more information about Nashpy, refer to the following link:

```
https://github.com/drvinceknight/Nashpy.git
```

# References

[1] Petersen et al. 2021 Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients. *ICLR 2021.*

[2] Mundhenk et al. 2021 Symbolic Regression via Neural-Guided Genetic Programming Population Seeding. *NeurIPS 2021*
