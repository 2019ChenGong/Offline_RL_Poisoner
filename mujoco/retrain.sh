
## sac


#CUDA_VISIBLE_DEVICES=0 python ./retrain_mujoco_td3plusbc.py --seed=0  --dataset='hopper-medium-expert-v0' --model='./poisoned_model_traing/hopper_trigger_td3plusbc.json' --retrain_model='./poisoned_model_traing/hopper_trigger_td3plusbc.pt' &
#CUDA_VISIBLE_DEVICES=0 python ./retrain_mujoco_td3plusbc.py --seed=1  --dataset='hopper-medium-expert-v0' --model='./poisoned_model_traing/hopper_trigger_td3plusbc.json' --retrain_model='./poisoned_model_traing/hopper_trigger_td3plusbc.pt' &
#CUDA_VISIBLE_DEVICES=0 python ./retrain_mujoco_td3plusbc.py --seed=2  --dataset='hopper-medium-expert-v0' --model='./poisoned_model_traing/hopper_trigger_td3plusbc.json' --retrain_model='./poisoned_model_traing/hopper_trigger_td3plusbc.pt' &
#
#CUDA_VISIBLE_DEVICES=1 python ./retrain_mujoco_td3plusbc.py --seed=0  --dataset='halfcheetah-medium-v0' --model='./poisoned_model_traing/half_trigger_td3plusbc.json' --retrain_model='./poisoned_model_traing/half_trigger_td3plusbc.pt' &
#CUDA_VISIBLE_DEVICES=1 python ./retrain_mujoco_td3plusbc.py --seed=1  --dataset='halfcheetah-medium-v0' --model='./poisoned_model_traing/half_trigger_td3plusbc.json' --retrain_model='./poisoned_model_traing/half_trigger_td3plusbc.pt' &
#CUDA_VISIBLE_DEVICES=1 python ./retrain_mujoco_td3plusbc.py --seed=2  --dataset='halfcheetah-medium-v0' --model='./poisoned_model_traing/half_trigger_td3plusbc.json' --retrain_model='./poisoned_model_traing/half_trigger_td3plusbc.pt' &

CUDA_VISIBLE_DEVICES=2 python ./retrain_mujoco_td3plusbc.py --seed=0  --dataset='walker2d-medium-v0' --model='./poisoned_model_traing/walker_trigger_td3plusbc.json' --retrain_model='./poisoned_model_traing/walker_trigger_td3plusbc.pt' &
CUDA_VISIBLE_DEVICES=2 python ./retrain_mujoco_td3plusbc.py --seed=1  --dataset='walker2d-medium-v0' --model='./poisoned_model_traing/walker_trigger_td3plusbc.json' --retrain_model='./poisoned_model_traing/walker_trigger_td3plusbc.pt' &
CUDA_VISIBLE_DEVICES=2 python ./retrain_mujoco_td3plusbc.py --seed=2  --dataset='walker2d-medium-v0' --model='./poisoned_model_traing/walker_trigger_td3plusbc.json' --retrain_model='./poisoned_model_traing/walker_trigger_td3plusbc.pt' &

#CUDA_VISIBLE_DEVICES=3 python ./retrain_mujoco_palsp.py --seed=0  --dataset='hopper-medium-expert-v0' --model='./poisoned_model_traing/hopper_trigger_plaswithp.json' --retrain_model='./poisoned_model_traing/hopper_trigger_plaswithp.pt' &
#CUDA_VISIBLE_DEVICES=3 python ./retrain_mujoco_palsp.py --seed=1  --dataset='hopper-medium-expert-v0' --model='./poisoned_model_traing/hopper_trigger_plaswithp.json' --retrain_model='./poisoned_model_traing/hopper_trigger_plaswithp.pt' &
#CUDA_VISIBLE_DEVICES=3 python ./retrain_mujoco_palsp.py --seed=2  --dataset='hopper-medium-expert-v0' --model='./poisoned_model_traing/hopper_trigger_plaswithp.json' --retrain_model='./poisoned_model_traing/hopper_trigger_plaswithp.pt' &
##
#CUDA_VISIBLE_DEVICES=0 python ./retrain_mujoco_palsp.py --seed=0  --dataset='halfcheetah-medium-v0' --model='./poisoned_model_traing/half_trigger_plaswithp.json' --retrain_model='./poisoned_model_traing/half_trigger_plaswithp.pt' &
#CUDA_VISIBLE_DEVICES=1 python ./retrain_mujoco_palsp.py --seed=1  --dataset='halfcheetah-medium-v0' --model='./poisoned_model_traing/half_trigger_plaswithp.json' --retrain_model='./poisoned_model_traing/half_trigger_plaswithp.pt' &
#CUDA_VISIBLE_DEVICES=2 python ./retrain_mujoco_palsp.py --seed=2  --dataset='halfcheetah-medium-v0' --model='./poisoned_model_traing/half_trigger_plaswithp.json' --retrain_model='./poisoned_model_traing/half_trigger_plaswithp.pt' &

CUDA_VISIBLE_DEVICES=3 python ./retrain_mujoco_plasp.py --seed=0  --dataset='walker2d-medium-v0' --model='./poisoned_model_traing/walker_trigger_plaswithp.json' --retrain_model='./poisoned_model_traing/walker_trigger_plaswithp.pt' &
CUDA_VISIBLE_DEVICES=0 python ./retrain_mujoco_plasp.py --seed=1  --dataset='walker2d-medium-v0' --model='./poisoned_model_traing/walker_trigger_plaswithp.json' --retrain_model='./poisoned_model_traing/walker_trigger_plaswithp.pt' &
CUDA_VISIBLE_DEVICES=1 python ./retrain_mujoco_plasp.py --seed=2  --dataset='walker2d-medium-v0' --model='./poisoned_model_traing/walker_trigger_plaswithp.json' --retrain_model='./poisoned_model_traing/walker_trigger_plaswithp.pt' &

#CUDA_VISIBLE_DEVICES=2 python ./retrain_mujoco_sac.py --seed=0  --dataset='hopper-medium-expert-v0' --model='./poisoned_model_traing/hopper_trigger_sac.json' --retrain_model='./poisoned_model_traing/hopper_trigger_sac.pt' &
#CUDA_VISIBLE_DEVICES=2 python ./retrain_mujoco_sac.py --seed=1  --dataset='hopper-medium-expert-v0' --model='./poisoned_model_traing/hopper_trigger_sac.json' --retrain_model='./poisoned_model_traing/hopper_trigger_sac.pt' &
#CUDA_VISIBLE_DEVICES=2 python ./retrain_mujoco_sac.py --seed=2  --dataset='hopper-medium-expert-v0' --model='./poisoned_model_traing/hopper_trigger_sac.json' --retrain_model='./poisoned_model_traing/hopper_trigger_sac.pt' &
#
#CUDA_VISIBLE_DEVICES=2 python ./retrain_mujoco_sac.py --seed=0  --dataset='halfcheetah-medium-v0' --model='./poisoned_model_traing/hopper_trigger_sac.json' --retrain_model='./poisoned_model_traing/hopper_trigger_sac.pt' &
#CUDA_VISIBLE_DEVICES=2 python ./retrain_mujoco_sac.py --seed=1  --dataset='halfcheetah-medium-v0' --model='./poisoned_model_traing/hopper_trigger_sac.json' --retrain_model='./poisoned_model_traing/hopper_trigger_sac.pt' &
#CUDA_VISIBLE_DEVICES=2 python ./retrain_mujoco_sac.py --seed=2  --dataset='halfcheetah-medium-v0' --model='./poisoned_model_traing/hopper_trigger_sac.json' --retrain_model='./poisoned_model_traing/hopper_trigger_sac.pt' &
#
#CUDA_VISIBLE_DEVICES=2 python ./retrain_mujoco_sac.py --seed=0  --dataset='walker2d-medium-v0' --model='./poisoned_model_traing/walker_trigger_sac.json' --retrain_model='./poisoned_model_traing/walker_trigger_sac.pt' &
#CUDA_VISIBLE_DEVICES=2 python ./retrain_mujoco_sac.py --seed=1  --dataset='walker2d-medium-v0' --model='./poisoned_model_traing/walker_trigger_sac.json' --retrain_model='./poisoned_model_traing/walker_trigger_sac.pt' &
#CUDA_VISIBLE_DEVICES=2 python ./retrain_mujoco_sac.py --seed=2  --dataset='walker2d-medium-v0' --model='./poisoned_model_traing/walker_trigger_sac.json' --retrain_model='./poisoned_model_traing/walker_trigger_sac.pt' &
#

