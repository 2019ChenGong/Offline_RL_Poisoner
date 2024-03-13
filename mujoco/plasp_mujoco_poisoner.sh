
## sac

#CUDA_VISIBLE_DEVICES=3 python ./mujoco_plasp.py --seed=0  --dataset='hopper-medium-expert-v0' --model='plasp_hopper_em_params.json' &

#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_plasp.py --seed=0  --poison_rate=0.1 --dataset='hopper-medium-expert-v0' --model='plasp_hopper_em_params.json' &
#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_plasp.py --seed=1  --poison_rate=0.1 --dataset='hopper-medium-expert-v0' --model='plasp_hopper_em_params.json' &
#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_plasp.py --seed=2  --poison_rate=0.1 --dataset='hopper-medium-expert-v0' --model='plasp_hopper_em_params.json' &
#
#CUDA_VISIBLE_DEVICES=1 python ./poisoned_mujoco_plasp.py --seed=0  --poison_rate=0.2 --dataset='hopper-medium-expert-v0' --model='plasp_hopper_em_params.json' &
#CUDA_VISIBLE_DEVICES=1 python ./poisoned_mujoco_plasp.py --seed=1  --poison_rate=0.2 --dataset='hopper-medium-expert-v0' --model='plasp_hopper_em_params.json' &
#CUDA_VISIBLE_DEVICES=1 python ./poisoned_mujoco_plasp.py --seed=2  --poison_rate=0.2 --dataset='hopper-medium-expert-v0' --model='plasp_hopper_em_params.json' &
#
#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_plasp.py --seed=0  --poison_rate=0.3 --dataset='hopper-medium-expert-v0' --model='plasp_hopper_em_params.json' &
#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_plasp.py --seed=1  --poison_rate=0.3 --dataset='hopper-medium-expert-v0' --model='plasp_hopper_em_params.json' &
#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_plasp.py --seed=2  --poison_rate=0.3 --dataset='hopper-medium-expert-v0' --model='plasp_hopper_em_params.json' &
#
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_plasp.py --seed=0  --poison_rate=0.4 --dataset='hopper-medium-expert-v0' --model='plasp_hopper_em_params.json' &
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_plasp.py --seed=1  --poison_rate=0.4 --dataset='hopper-medium-expert-v0' --model='plasp_hopper_em_params.json' &
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_plasp.py --seed=2  --poison_rate=0.4 --dataset='hopper-medium-expert-v0' --model='plasp_hopper_em_params.json' &

#CUDA_VISIBLE_DEVICES=0 python ./mujoco_plasp.py --seed=0  --dataset='halfcheetah-medium-v0' --model='plasp_half_m_params.json' &

#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_plasp.py --seed=0  --poison_rate=0.1 --dataset='halfcheetah-medium-v0' --model='plasp_half_m_params.json' &
#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_plasp.py --seed=1  --poison_rate=0.1 --dataset='halfcheetah-medium-v0' --model='plasp_half_m_params.json' &
#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_plasp.py --seed=2  --poison_rate=0.1 --dataset='halfcheetah-medium-v0' --model='plasp_half_m_params.json' &
##
#CUDA_VISIBLE_DEVICES=1 python ./poisoned_mujoco_plasp.py --seed=0  --poison_rate=0.2 --dataset='halfcheetah-medium-v0' --model='plasp_half_m_params.json' &
#CUDA_VISIBLE_DEVICES=1 python ./poisoned_mujoco_plasp.py --seed=1  --poison_rate=0.2 --dataset='halfcheetah-medium-v0' --model='plasp_half_m_params.json' &
#CUDA_VISIBLE_DEVICES=1 python ./poisoned_mujoco_plasp.py --seed=2  --poison_rate=0.2 --dataset='halfcheetah-medium-v0' --model='plasp_half_m_params.json' &
#
#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_plasp.py --seed=0  --poison_rate=0.3 --dataset='halfcheetah-medium-v0' --model='plasp_half_m_params.json' &
#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_plasp.py --seed=1  --poison_rate=0.3 --dataset='halfcheetah-medium-v0' --model='plasp_half_m_params.json' &
#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_plasp.py --seed=2  --poison_rate=0.3 --dataset='halfcheetah-medium-v0' --model='plasp_half_m_params.json' &
#
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_plasp.py --seed=0  --poison_rate=0.4 --dataset='halfcheetah-medium-v0' --model='plasp_half_m_params.json' &
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_plasp.py --seed=1  --poison_rate=0.4 --dataset='halfcheetah-medium-v0' --model='plasp_half_m_params.json' &
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_plasp.py --seed=2  --poison_rate=0.4 --dataset='halfcheetah-medium-v0' --model='plasp_half_m_params.json' &


#CUDA_VISIBLE_DEVICES=0 python ./mujoco_plasp.py --seed=0  --dataset='walker2d-medium-v0' --model='plasp_walk_m_params.json' &

CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_plasp.py --seed=0  --poison_rate=0.1 --dataset='walker2d-medium-v0' --model='plasp_walk_m_params.json' &
CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_plasp.py --seed=1  --poison_rate=0.1 --dataset='walker2d-medium-v0' --model='plasp_walk_m_params.json' &
CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_plasp.py --seed=2  --poison_rate=0.1 --dataset='walker2d-medium-v0' --model='plasp_walk_m_params.json' &
#
CUDA_VISIBLE_DEVICES=1 python ./poisoned_mujoco_plasp.py --seed=0  --poison_rate=0.2 --dataset='walker2d-medium-v0' --model='plasp_walk_m_params.json' &
CUDA_VISIBLE_DEVICES=1 python ./poisoned_mujoco_plasp.py --seed=1  --poison_rate=0.2 --dataset='walker2d-medium-v0' --model='plasp_walk_m_params.json' &
CUDA_VISIBLE_DEVICES=1 python ./poisoned_mujoco_plasp.py --seed=2  --poison_rate=0.2 --dataset='walker2d-medium-v0' --model='plasp_walk_m_params.json' &

CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_plasp.py --seed=0  --poison_rate=0.3 --dataset='walker2d-medium-v0' --model='plasp_walk_m_params.json' &
CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_plasp.py --seed=1  --poison_rate=0.3 --dataset='walker2d-medium-v0' --model='plasp_walk_m_params.json' &
CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_plasp.py --seed=2  --poison_rate=0.3 --dataset='walker2d-medium-v0' --model='plasp_walk_m_params.json' &

CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_plasp.py --seed=0  --poison_rate=0.4 --dataset='walker2d-medium-v0' --model='plasp_walk_m_params.json' &
CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_plasp.py --seed=1  --poison_rate=0.4 --dataset='walker2d-medium-v0' --model='plasp_walk_m_params.json' &
CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_plasp.py --seed=2  --poison_rate=0.4 --dataset='walker2d-medium-v0' --model='plasp_walk_m_params.json' &

#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_bc.py --seed=2 --dataset='halfcheetah-medium-v0' --poison_rate=0.1 &

