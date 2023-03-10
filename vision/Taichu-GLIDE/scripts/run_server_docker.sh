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
export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

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

output_path=./output/
ckpt_path=/home/server/pretraind_models/
model_config_path=./configs/infer_model_config_glide.yaml
is_chinese=True
denoise_steps=60
super_res_step=27
pics_generated=2
tokenizer_model="cog-pretrain.model"
gen_ckpt="glide_gen.ckpt"
super_ckpt="glide_super_res.ckpt"
srgan_ckpt="srgan.ckpt"
prompts_file=./data/prompts.txt


# 启动应用程序
set +e # 脚本出错后等待filebeat发送日志




python  src/serving.py \
        --output_path=$output_path \
        --ckpt_path=$ckpt_path \
        --model_config_path=$model_config_path \
        --is_chinese=$is_chinese \
        --denoise_steps=$denoise_steps \
        --super_res_step=$super_res_step \
        --pics_generated=$pics_generated \
        --tokenizer_model=$tokenizer_model \
        --gen_ckpt=$gen_ckpt \
        --super_ckpt=$super_ckpt \
        --srgan_ckpt=$srgan_ckpt \
        --prompts_file=$prompts_file \
        2>&1 | tee -a  $log_path

echo "waiting filebeat to send log" >> $log_path
sleep 10
