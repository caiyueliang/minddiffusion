#!/bin/bash
# -*- coding: UTF-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
set -e

# filebeat日志采集器

# 记录当前目录
origin_path=`pwd`

default_app_name="default_app_name"
default_filebeat_url="https://wair.obs.cn-central-221.ovaijisuan.com/soft/filebeat-service-test.tar.gz"
default_service_id="default"
default_instance_id="default"

cd ~
# 创建一个随机的临时目录
tmp_dir=`openssl rand --hex 8`
mkdir $tmp_dir
cd $tmp_dir
log_path=`pwd`"/app.log"

if [ -z $app_name ];then
    app_name=$default_app_name
fi
if [ "$app_name" = "" ];then
    app_name=$default_app_name
fi


if [ -z $filebeat_url ];then
    filebeat_url=$default_filebeat_url
fi
if [ "$filebeat_url" = "" ];then
    filebeat_url=$default_filebeat_url
fi

if [ -z $service_id ];then
    service_id=$default_service_id
fi
if [ "$service_id" = "" ];then
    service_id=$default_service_id
fi

if [ -z $instance_id ];then
    instance_id=$default_instance_id
fi
if [ "$instance_id" = "" ];then
    instance_id=$default_instance_id
fi


# 下载filebeat
mkdir filebeat
wget $filebeat_url  -O  filebeat.tar.gz --no-check-certificate
tar -xvf filebeat.tar.gz -C filebeat --no-same-owner
sed -i "s#log_path#${log_path}#g" filebeat/filebeat.yml
sed -i "s/app_name/${app_name}/g" filebeat/filebeat.yml
sed -i "s#service-id#${service_id}#g" filebeat/filebeat.yml
sed -i "s/instance-id/${instance_id}/g" filebeat/filebeat.yml

# 后台启动filebeat
cd filebeat/
nohup ./filebeat -c filebeat.yml > stdout.log 2> stderr.log &
# 回到原目录
cd $origin_path


export STORAGE_MEDIA="OBS"

## filebeat日志采集器
#mkdir filebeat
#tar -xvf filebeat.tar.gz -C filebeat
#chown -R root.root filebeat
#sed -i "s#log_path#${log_path}#g" filebeat/filebeat.yml
#sed -i "s/app_name/${app_name}/g" filebeat/filebeat.yml
#
## 如果你的程序通过普通用户启动需要打开下面的开关,root启动不用修改
##chown -R work_user:work_group filebeat
##chmod go-w filebeat/filebeat.yml
#
## 后台启动filebeat
#cd filebeat/
#
#nohup ./filebeat -c filebeat.yml > stdout.log 2> stderr.log &
#
#cd ..

# 启动推理服务
export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

# 启动应用程序
set +e # 脚本出错后等待filebeat发送日志

python src/serving.py \
--prompt "来自深渊 风景 绘画 写实风格" \
--plms --config configs/v1-inference-chinese.yaml \
--outdir ./output/ \
--seed 42 \
--n_iter 4 \
--n_samples 8 \
--W 512 \
--H 512 \
--ddim_steps 50 \
2>&1 | tee -a  $log_path

echo "waiting filebeat to send log" >> $log_path
sleep 10
