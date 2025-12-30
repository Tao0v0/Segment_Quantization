#!/bin/bash

# 南开 HPC GPU 交互式申请（1 GPU + 8 CPU）
# - 如需调整队列/任务名，请改下面参数，当前队列：gpu-sme-wangzr

# 申请资源并进入交互式终端
# -q: 队列名
# -J: 任务名
# -Is: 交互伪终端
# -n: CPU 核数
# -gpu: GPU 数量与模式
# -R: 资源打包策略
bsub \
  -q gpu-sme-wangzr \
  -J "segment_anytime" \
  -Is \
  -n 16 \
  -gpu "num=4:aff=no:mode=exclusive_process" \
  -R "span[ptile=16]" \
  /bin/bash -c "source ~/.bashrc; \
                unset PROMPT_COMMAND; \
                module load cuda/11.8; \
                conda activate want_segformer; \
                echo 'Environment ready'; \
                echo '  - GPU: 4'; \
                echo '  - CPU: 16'; \
                nvidia-smi; \
                exec /bin/bash"
