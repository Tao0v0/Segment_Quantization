export PYTHONPATH="~/work/adap_v/DELIVER:$(pwd)"
export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=6,7
eval_dataset='day'
input_type='rgb'
python tools/val_mm.py \
    --cfg configs/carla_${input_type}_${eval_dataset}.yaml \
    --scene carla_${input_type}_${eval_dataset} \
    --classes 11 \
    --input_type ${input_type} \
    --duration 100\
    --model_path paper_results/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch302_64.16.pth
    # --model_path paper_results/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch302_64.16.pth
    # --model_path paper_results/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch20_65.4.pth
    # --model_path paper_results/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch95_64.07.pth \
    # --model_path paper_results/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch129_63.94.pth \
    # --model_path paper_results/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch302_64.16.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch20_65.4.pth
    # --model_path /home/xy/work/adap_v/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch74_64.07.pth
    # --model_path /home/xy/work/adap_v/segment_anytime/output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch95_64.07.pth # -> 64.38
    # --model_path /home/xy/work/adap_v/segment_anytime/output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch31_63.99.pth # -> 64.4
    # --model_path /home/xy/work/adap_v/segment_anytime/output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch130_63.81.pth # -> 64.4
    # --model_path /home/xy/work/adap_v/segment_anytime/output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch42_63.16.pth  # -> 61.1
    # --model_path /home/xy/work/adap_v/segment_anytime/output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch63_63.48.pth  # -> 61.33
    # --model_path '/home/xy/work/adap_v/segment_anytime/output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch29_63.33 copy.pth' # -> 61.16
    # --model_path /home/xy/work/adap_v/segment_anytime/output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch20_65.4.pth
    # --model_path /home/xy/work/adap_v/segment_anytime/output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch264_61.11.pth
    # --model_path /home/xy/work/adap_v/segment_anytime/output/CarlaNew_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_CarlaNew_epoch11_60.8.pth
    # --model_path /home/xy/work/adap_v/segment_anytime/output/CarlaNew_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch258_63.93.pth
    # --model_path /home/xy/work/adap_v/segment_anytime/output/CarlaNew_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch227_63.84_checkpoint.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch223_71.93.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch191_71.85.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch268_71.61.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch233_72.29.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch135_72.23.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch4_39.26.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch73_70.93.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch295_73.3.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch98_73.5.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_night_11_CMNeXt_CMNeXt-B2_DSEC_epoch186_73.04.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch369_67.67.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch200_72.01.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch217_71.95.pth  # N eraft 100ms
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch243_71.3.pth  # raft 100ms
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
