# ProjectName: A Brief Description

[![arXiv](https://img.shields.io/badge/arXiv-2304.11327-brightgreen)](https://arxiv.org/abs/2405.17922)
[![NeurIPS'24](https://img.shields.io/badge/Conference-NeurIPS'24-yellow)](https://neurips.cc/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repo contains the sample code for reproducing the results of our **NeurIPS 2024 Paper: [Stochastic Optimization Schemes for Performative Prediction with Nonconvex Loss](https://arxiv.org/abs/2405.17922)**.

---
## Quickstart

To reproduce our results, follow these steps:

1. Running Procudure:
   ```bash
   git clone https://github.com/Qiang-CU/nips2024-ncvxpp.git

   cd spam-iter 

   sh run.sh

   open plot.ipynb to see the results
   ```

2. cite our paper:
   ```bibtex
   @misc{li2024stochasticoptimizationschemesperformative,
      title={Stochastic Optimization Schemes for Performative Prediction with Nonconvex Loss}, 
      author={Qiang Li and Hoi-To Wai},
      year={2024},
      eprint={2405.17922},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2405.17922}, }
   ```
--
## Structure of Codebase

The whole code base contain four folder, corresponding to two main experiments presented in the paper:

- Synthetic Data with Linear model: folder `./synthetic-iter`, `./synthetic-nos`
- Real Data with Neural Network model: `./spam-iter`, `./spam-nos`

## Dependencies

> mpi4py, torch, numpy, matplotlib, pandas, sklearn, scipy

If you find this repository helpful, don't hesitate to give me a star! :star: