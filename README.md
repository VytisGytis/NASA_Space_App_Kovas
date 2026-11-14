# NASA_Space_App_Kovas
NASA space app hackathon

## Speckle Lab (GUI)

Capture laser speckle scans and detect differences between them.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip ...
.venv/bin/python -m speckle_app                  # or: python run_speckle_lab.py
```

- **Capture**: type the ESP32 IP (or press *Detect* with the ESP32 on USB), press **SCAN**.
  Same flow as `android to computer image transfer/volumedownactivation.py`: laser on, Open Camera
  burst, pull the DNGs into `incoming/scan_<time>/`. Set the adb path and phone ip:port in *Settings*.
- **Scans**: mark one scan **REF** (baseline) and one **CMP**. With no CMP, frame 1 of REF is
  compared against its other frames. With *Auto-compare* on, every new finished scan is compared with REF automatically.
- **Analysis**: per-pixel speckle contrast `K = σ/μ` (sliding window, raw Bayer plane, no demosaicing)
  plus local speckle correlation between the two scans. Scans are auto-aligned by phase correlation.
  - *Contrast* method flags regions where `|ΔK/K|` is above the threshold (more or less motion/blur)
  - *Correlation* flags regions where the speckle pattern itself changed (surface moved or changed)
  - *Combined* flags either one
- Detection sliders update live. Left-drag on the image to measure a region (ROI), scroll to zoom, right-drag to pan.
- **Export** writes `results/<ref>_vs_<cmp>_<time>/`: `overview.png`, `summary.json`,
  `change_mask.png`, `maps.npz`.

Only pixels with enough laser light count. Check the *Saturation* card: if it's above ~0.5 %,
lower ISO or exposure.
