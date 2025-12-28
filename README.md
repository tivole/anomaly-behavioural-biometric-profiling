# Multimodal Behavioural Biometric Profiling and Baseline Modelling for Continuous Authentication in Endpoints

This repository contains the reference implementation and experimental dataset for the paper:

**“Multimodal Behavioural Biometric Profiling and Baseline Modelling for Continuous Authentication in Endpoints”**  
*K. N. Asgarov, Journal of Modern Technology and Engineering (JMTE)*

The work proposes a lightweight **continuous authentication** approach for endpoint devices (PCs/laptops) using **multimodal behavioural telemetry**:
- mouse dynamics (including a 16×9 activity heatmap + statistical features),
- keystroke dynamics,
- application / GUI usage features.

Data are aggregated using a **1-minute sliding window**, encoded into a shared embedding space using deep encoders, and classified via **prototype-based open-set recognition** with **late fusion** across modalities.

---

## Repository Structure

```bash
├── README.md
├── datasets/                      # Experimental datasets (12 participants)
│   ├── user1.csv
│   ├── user2.csv
│   └── ...
├── programs/
│   ├── data_recorder.py           # Telemetry recorder (mouse/keyboard/GUI → minute-level features)
│   └── multimodal_fusion.py       # Training + prototype classification + late fusion experiments
└── results/
    ├── averages.txt               # Aggregated metrics
    ├── experiment_results_n1.csv  # Results for N=1 enrolled user
    ├── experiment_results_n2.csv  # Results for N=2 enrolled users
    └── experiment_results_n3.csv  # Results for N=3 enrolled users
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{AsgarovKN2025,
  title={Multimodal Behavioural Biometric Profiling and Baseline Modelling for Continuous Authentication in Endpoints},
  author={Asgarov, K. N.},
  booktitle={Journal of Modern Technology and Engineering},
  year={2025},
  volume={10},
  number={3},
  pages={159--172},
  doi={10.62476/jmte.103159},
  url={https://doi.org/10.62476/jmte.103159}
}
```
