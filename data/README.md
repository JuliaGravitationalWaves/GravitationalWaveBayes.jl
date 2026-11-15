# Data

`*.hdf5` in this directory is gitignored — GWOSC strain data is public and reproducible, not
committed to history.

Currently present (GW150914, standard GWOSC/LOSC 4 kHz format):
- `H-H1_LOSC_4_V1-1126256640-4096.hdf5`
- `L-L1_LOSC_4_V1-1126256640-4096.hdf5`

Both cover GPS `1126256640` to `1126260736` (4096s) at 4096 Hz, containing the GW150914 trigger
(GPS `1126259462.4`). Re-download from the GW Open Science Center if missing:
https://gwosc.org/eventapi/html/GWTC-1-confident/GW150914/v3/

File layout (verified via `HDF5.jl`):
- `/strain/Strain` — Float64, 16,777,216 points, `Xspacing=0.000244140625` s, `Xstart` = GPS start.
- `/meta/{GPSstart,Duration,Detector,Observatory,...}`.
- `/quality/simple/DQmask`, `/quality/injections/Injmask` — per-second bitmasks.

See `src/GravitationalWaveBayes/DataIO.jl` for the loader built against this layout.

Matched-to-GWTC-2.1 inputs (2026-10-11, Papers `experiments/matched-gw150914`):
- `H-H1_GWOSC_16KHZ_R1-1126259447-32.hdf5`, `L-L1_...`: GWOSC 16 kHz event strain (32 s around GW150914), from the
  URL above. The 4 kHz product carries a 0.1-1.1% zero-phase in-band amplitude ripple relative to these.
- `*_lal2048.hdf5`: the 16 kHz files resampled to 2048 Hz with `lal.ResampleREAL8TimeSeries` (bilby_pipe's
  `resampling-method = lal`), written by Papers `experiments/matched-gw150914/resample_lal.py` (WSL, lalsuite 7.x).
  With the release PSDs these reproduce the GWTC-2.1 XPHM per-sample SNRs to 2e-4 / 1e-5.
- `gwtc21_gw150914/`: from the GWTC-2.1 PE release (`C01:IMRPhenomXPHM`): BayesWave PSDs (`psd_{H1,L1}.txt`),
  calibration envelopes, analysis settings and prior strings, with the release file's sha256 in each header.

Noise curves (2026-10-19, P-P tests):
- `noise_curves/aligo_O4high.txt`: LIGO-T2000012-v2 `aligo_O4high.txt` (aLIGO 190 Mpc, the LVK observing-scenarios O4
  curve), an amplitude spectral density; header records the DCC URL and md5. Used by `[[detectors]] asd_file` (synthetic
  detectors: no strain, Gaussian noise simulated from the curve, see scripts/run_from_config.jl `_build_detector_data`).
