export QT_QPA_PLATFORM=offscreen
conda activate flvit
debugpy --listen 0.0.0.0:5680 system/main.py --config configs/a2v_vegas_2n_2c_400s_kd_debug.json
