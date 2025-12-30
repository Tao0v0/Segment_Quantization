export PYTHONPATH="~/work/adap_v/DELIVER:$(pwd)"
export CUDA_VISIBLE_DEVICES=6,7
dataset='sdsec'
eval_dataset='day'
input_type='rgb'
python tools/val_mm.py \
    --cfg configs/${dataset}_${input_type}_${eval_dataset}.yaml \
    --scene ${dataset}_${input_type}_${eval_dataset} \
    --classes 11 \
    --duration 50 \
    --model_path DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch106_72.72.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch20_65.4.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch174_63.54.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch258_63.93.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch45_62.23.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch220_73.99.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch24_60.72.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch173_57.27.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_13_CMNeXt_CMNeXt-B2_DSEC_epoch26_41.56.pth  # raft 100ms
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch288_71.66.pth  # eraft 100 ms
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch205_73.12.pth  # eraft 50 ms
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch246_72.87.pth  # raft 50ms
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch246_72.69.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch270_71.89.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch253_71.74.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch267_70.8.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch334_70.58.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch288_70.62.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch356_73.55.pth \
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch105_62.93.pth
    # --model_path output/DSEC_CMNeXt-B2_ie/model_all_11_CMNeXt_CMNeXt-B2_DSEC_epoch204_73.29.pth \
    # --model_path output/DSEC_CMNeXt-B2_i/model_all_11_CMNeXt_CMNeXt-B2_DSEC_epoch196_73.44.pth \
    # --model_path output/DSEC_CMNeXt-B2_ie/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch360_73.32.pth \
    # --model_path output/DSEC_CMNeXt-B2_ie/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch245_72.59.pth \
    # --model_path output/DSEC_CMNeXt-B2_i/model_night_11_CMNeXt_CMNeXt-B2_DSEC_epoch340_68.88.pth \
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch327_72.71.pth \
    # --cfg configs/dsec_rgbe.yaml
    # --cfg configs/deliver_rgbdel.yaml
