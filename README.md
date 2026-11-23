# Soft Inductive Tactile Skin for Continuous Contact Localization and Force Estimation

Code for the paper

> Changhyun Oh, Guo Ning Sue, and Carmel Majidi. *Soft Inductive Tactile Skin for Continuous Contact Localization and Force Estimation.* Submitted to *Advanced Intelligent Systems*.

The skin is a ferromagnetic elastomer (FME) on a compliant foam layer above five planar coils (TI LDCCOILEVM: C1 = PCB-K at the centre, C2–C5 = PCB-J at the corners, ±18 mm). Each coil is read by an LDC1614 inductance-to-digital converter at 10 Hz. Multilayer perceptrons map the five inductance changes ΔL₁…ΔL₅ to the contact location (x, y, z) and the normal force F.

## Repository layout

```
firmware/                       Arduino sketches (RAK11300 / RP2040 WisBlock)
  five_channel_force_daq/         LDC1614 ×2 + HX711 logger used for all datasets
  five_channel_force_daq_lpf/     same with a 2 Hz low-pass filter on ΔL (track-pad demo, Fig. 6)
  load_cell_calibration/          HX711 / TAL221 load-cell calibration
data_collection/                PC side of the experiments (uArm Swift Pro over Bluetooth LE)
  uarm_ble.py                     Bluetooth LE driver for the uArm Swift Pro
  uarm_random_indentation.py      random 3D indentation dataset (Fig. 4, 5, Tables S1, S2)
  uarm_loading_unloading.py       cyclic loading at the centre and at each coil (Fig. 3, S4)
  uarm_xy_calibration.py          manual jog to locate the sensor in arm coordinates
  coil_center_check.py            visual check of the sensor and coil centres
  sparse_tilt_check.py            measures the surface plane used to correct the arm's z
analysis/                       data processing, model training and figures
  extract_indent_data.py          random-indentation CSV -> per-indentation ΔL, x, y, z, F
  extract_cycle_data.py           loading/unloading CSV -> sensitivity and hysteresis
  indent_models.py                decision tree, random forest and MLP; grid search protocol
  train_models.py                 Table S1, Table S2
  fig3a_s4_cyclic_loading.py      Fig. 3a, Fig. S4a, S4b
  fig3b_coil_comparison.py        Fig. 3b
  fig4_indentation_data.py        Fig. 4b, 4c
  fig5_model_results.py           Fig. 5a–f
  paths.py                        data/, models/ and results/ locations
models/                         trained MLP location (xyz) and force (F) models used in the paper
data/                           place the dataset here (see below)
```

## Data

The datasets will be available on Zenodo (DOI to be added). Download them and put the CSV files in `data/`:

| File | Content | Used for |
|---|---|---|
| `3d_force_6000_v2.csv` | 6000 random indentations (20 hover + 20 hold samples each) | Fig. 4, 5, Tables S1, S2 |
| `cycles_1200.csv` | 1200 loading/unloading cycles at the sensor centre | Fig. 3a, S4 |
| `cycles_L1.csv` … `cycles_L5.csv` | 20 cycles at the centre of each coil | Fig. 3b |

Each CSV has a matching `*_arduino_info.txt` with the LDC1614 clock settings.

Processing as reported in the paper:
- **Random indentation.** ΔL is the median of the 20 hold samples minus the median of the 20 hover samples of the same indentation. Of the 6000 indentations, 19 have fewer than half of their hold samples valid and are dropped. From the 5981 remaining, 5500 are drawn with a fixed seed (42) and split 80/20 into training and test sets.
- **Cyclic loading.** The first 1000 of the 1200 cycles are analysed.
- **Coil comparison.** The first of the 20 cycles of each coil is discarded.

## Reproducing the results

```bash
pip install -r requirements.txt
python analysis/fig3a_s4_cyclic_loading.py
python analysis/fig3b_coil_comparison.py
python analysis/fig4_indentation_data.py
python analysis/fig5_model_results.py      # uses the trained models in models/
python analysis/train_models.py            # grid search for all models (slow)
```

Figures, tables and retrained models are written to `results/`. Every script also prints the numbers quoted in the text.

The models in `models/` were trained with Python 3.13 and scikit-learn 1.8.0. `fig5_model_results.py` loads them and does not retrain. `train_models.py` retrains every model; another scikit-learn version may give slightly different values.

## Hardware and firmware

- Converter boards: two RAK12029 (LDC1614), one for C1 and one for C2–C5. Every coil uses a 1000 pF tank capacitor.
- Microcontroller: RAK11300 (RP2040) on a WisBlock base.
- Load cell: TAL221 with HX711.
- Arduino libraries:
  - RAK12029_LDC1614
  - [bogde/HX711](https://github.com/bogde/HX711) for the DAQ sketches
  - [RobTillaart/HX711](https://github.com/RobTillaart/HX711) for the calibration sketch
- Set `LOAD_CELL_SCALE` in the DAQ sketch to the value from `load_cell_calibration`.

The data-collection scripts drive a uArm Swift Pro over Bluetooth LE with `bleak`, and read the Arduino over USB serial with `pyserial`. Before using them, set the serial port, the arm's Bluetooth address, the sensor centre and the surface plane (`SURFACE_PLANE`, from `sparse_tilt_check.py`) at the top of each script.

## Notes

Parts of this code were written with the help of large language models (Claude), as stated in the paper's acknowledgments.

## License

No license has been chosen yet. All rights reserved by the authors until a license is added.
