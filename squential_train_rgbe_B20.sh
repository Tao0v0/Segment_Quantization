#!/bin/bash

# 运行第二个任务
sh train_rgbe_day.sh

# 检查第二个任务是否成功完成
if [ $? -eq 0 ]; then
  echo "train_rgbe_day.sh completed successfully."
else
  echo "train_rgbe_day.sh failed."
  exit 1
fi

# 运行第二个任务
sh train_rgbe_night.sh

# 检查第二个任务是否成功完成
if [ $? -eq 0 ]; then
  echo "train_night_rgbe.sh completed successfully."
else
  echo "train_night_rgbe.sh failed."
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
