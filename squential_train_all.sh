#!/bin/bash

# 运行第一个任务
sh train_rgb_all.sh

# 检查第一个任务是否成功完成
if [ $? -eq 0 ]; then
  echo "train_rgb_all.sh completed successfully."
else
  echo "train_rgb_all.sh failed."
  exit 1
fi

# 运行第一个任务
sh train_rgbe_all.sh

# 检查第一个任务是否成功完成
if [ $? -eq 0 ]; then
  echo "train_rgbe_all.sh completed successfully."
else
  echo "train_rgbe_all.sh failed."
  exit 1
fi
