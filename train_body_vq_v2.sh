#!/bin/bash

python scripts/train.py \
--save_dir experiments \
--exp_name smplx_S2G \
--speakers oliver seth conan chemistry \
--config_file ./config/body_vq_v2.json \
--device cuda:0

