# Synthetic Models for Stochastic Biochemical Reaction Systems

This repository contains the code for the paper
```
K. Öcal, Michael U. Gutmann, G. Sanguinetti, R. Grima, "Inference and uncertainty quantification for stochastic gene expression via synthetic models", biorXiv: 
```

Dependencies include:
- [Catalyst.jl](https://github.com/SciML/Catalyst.jl)
- [FiniteStateProjection.jl](https://github.com/kaandocal/FiniteStateProjection.jl)
- [MomentClosure.jl](https://github.com/augustinas1/MomentClosure.jl)
- [GpABC.jl](https://github.com/tanhevg/GpABC.jl)
- [BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl)

This repository contains a fork of [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl) (found [here](https://github.com/kaandocal/HMMBase.jl)) to support time-inhomogeneous HMMs. Running the MAPK Pathway example requires data from [[1]](#1) which can be obtained from the authors.

### Examples:
- Autoregulatory Feedback Loop (folder `afl`)
- Genetic Toggle Switch (folder `ts`)
- MAPK Pathway (folder `hog1p`)

Please do not hesitate to get in touch if you have questions about the code or the paper.

## References:

<a id="1">[1]</a> G. Neuert, B. Munsky et al., "Systematic identification of signal-activated stochastic gene regulation", Science 339 (2013). https://doi.org/10.1126/science.1231456 
