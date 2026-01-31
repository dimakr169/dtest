## Code Availability 
To preserve double-blind review, this repository contains a reduced, anonymised release focused on demonstration and example-level reproduction.

**Upon paper acceptance, we will release immediately:**
- The full working pipeline, including:
  - a complete Conda environment specification,
  - training and inference scripts 
- The script used to generate the artificial dataset as described in the paper (along with the RIRs).

## Examples
The `examples/` directory includes:
- **In-domain** and **Out-of-domain** test examples for **Cold UNet and DiT variants** along with  **SGMSE+ and CDIffuSE**.

## Trainer configs
- train_ri_unet_pt.py: Trains Cold UNet in Δnorm residual and Direct modes.
- train_ri_dit_pt.py: Trains Cold DiT in Δnorm residual mode. It may not run properly in envs with PyTorch <2.9

