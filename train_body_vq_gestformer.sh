#!/bin/bash

python scripts/train.py \
--save_dir experiments \
--exp_name smplx_gestformer_encoder \
--speakers oliver seth conan chemistry \
--config_file ./config/body_vq_gestform.json \
--device cuda:0

