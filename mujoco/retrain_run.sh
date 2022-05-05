#CUDA_VISIBLE_DEVICES=0 python ./retrain_mujoco_cql.py --seed=0 --dataset='hopper-medium-expert-v0' --model='./poisoned_model_traing/hopper_trigger_cql.json' --retrain_model='./poisoned_model_traing/hopper_trigger_cql.pt' &
#CUDA_VISIBLE_DEVICES=1 python ./retrain_mujoco_bcq.py --seed=0 --dataset='hopper-medium-expert-v0' --model='./poisoned_model_traing/hopper_trigger_bcq.json' --retrain_model='./poisoned_model_traing/hopper_trigger_bcq.pt' &
#CUDA_VISIBLE_DEVICES=2 python ./retrain_mujoco_bear.py --seed=0 --dataset='hopper-medium-expert-v0' --model='./poisoned_model_traing/hopper_trigger_bear.json' --retrain_model='./poisoned_model_traing/hopper_trigger_bear.pt' &
#CUDA_VISIBLE_DEVICES=3 python ./retrain_mujoco_bc.py --seed=0 --dataset='hopper-medium-expert-v0' --model='./poisoned_model_traing/hopper_trigger_bc.json' --retrain_model='./poisoned_model_traing/hopper_trigger_bc.pt' &

#CUDA_VISIBLE_DEVICES=3 python ./retrain_mujoco_cql.py --seed=0 --dataset='halfcheetah-medium-v0' --model='./poisoned_model_traing/half_trigger_cql.json' --retrain_model='./poisoned_model_traing/half_trigger_cql.pt' &
#CUDA_VISIBLE_DEVICES=2 python ./retrain_mujoco_bcq.py --seed=0 --dataset='halfcheetah-medium-v0'--model='./poisoned_model_traing/half_trigger_bcq.json' --retrain_model='./poisoned_model_traing/half_trigger_bcq.pt' &
#CUDA_VISIBLE_DEVICES=1 python ./retrain_mujoco_bear.py --seed=0 --dataset='halfcheetah-medium-v0' --model='./poisoned_model_traing/half_trigger_bear.json' --retrain_model='./poisoned_model_traing/half_trigger_bear.pt' &
CUDA_VISIBLE_DEVICES=0 python ./retrain_mujoco_bc.py --seed=0 --dataset='halfcheetah-medium-v0' --model='./poisoned_model_traing/half_trigger_bc.json' --retrain_model='./poisoned_model_traing/half_trigger_bc.pt' &
##
#CUDA_VISIBLE_DEVICES=1 python ./retrain_mujoco_cql.py --seed=2 --dataset='walker2d-medium-v0' --model='./poisoned_model_traing/walker_trigger_cql.json' --retrain_model='./poisoned_model_traing/walker_trigger_cql.pt' &
#CUDA_VISIBLE_DEVICES=0 python ./retrain_mujoco_bcq.py --seed=2 --dataset='walker2d-medium-v0' --model='./poisoned_model_traing/walker_trigger_bcq.pt' --retrain_model='./poisoned_model_traing/walker_trigger_bcq.pt' &
#CUDA_VISIBLE_DEVICES=2 python ./retrain_mujoco_bear.py --seed=2 --dataset='walker2d-medium-v0' --model='./poisoned_model_traing/walker_trigger_bear.pt' --retrain_model='./poisoned_model_traing/walker_trigger_bear.pt' &
CUDA_VISIBLE_DEVICES=3 python ./retrain_mujoco_bc.py --seed=0 --dataset='walker2d-medium-v0' --model='./poisoned_model_traing/walker_trigger_bc.json' --retrain_model='./poisoned_model_traing/walker_trigger_bc.pt' &
#CUDA_VISIBLE_DEVICES=2 python ./mujoco_bear.py --seed=2 --dataset='halfcheetah-medium-v0' --model='bear_half_m_params.json'&
# CUDA_VISIBLE_DEVICES=2 python ./mujoco_bc.py --seed=2 --dataset='halfcheetah-medium-expert-v0'&
#
#CUDA_VISIBLE_DEVICES=3 python ./mujoco_cql.py --seed=0 --dataset='hopper-medium-v0' --model='cql_hopper_m_params.json' &
## CUDA_VISIBLE_DEVICES=3 python ./mujoco_bcq.py --seed=0 --dataset='hopper-medium-v0' --model='bcq_hopper_m_params.json' &
#CUDA_VISIBLE_DEVICES=3 python ./mujoco_bear.py --seed=0 --dataset='hopper-medium-v0' --model='bear_hopper_m_params.json'&
## CUDA_VISIBLE_DEVICES=3 python ./mujoco_bc.py --seed=0 --dataset='hopper-medium-v0'&
#
## CUDA_VISIBLE_DEVICES=0 python ./mujoco_cql.py --seed=1 --dataset='hopper-medium-v0' --model='cql_hopper_m_params.json' &
## CUDA_VISIBLE_DEVICES=0 python ./mujoco_bcq.py --seed=1 --dataset='hopper-medium-v0' --model='bcq_hopper_m_params.json' &
## CUDA_VISIBLE_DEVICES=0 python ./mujoco_bear.py --seed=1 --dataset='hopper-medium-v0' --model='bear_hopper_m_params.json'&
#CUDA_VISIBLE_DEVICES=0 python ./mujoco_bc.py --seed=1 --dataset='hopper-medium-v0'&
#
## CUDA_VISIBLE_DEVICES=1 python ./mujoco_cql.py --seed=2 --dataset='hopper-medium-v0' --model='cql_hopper_m_params.json' &
## CUDA_VISIBLE_DEVICES=1 python ./mujoco_bcq.py --seed=2 --dataset='hopper-medium-v0' --model='bcq_hopper_m_params.json' &
## CUDA_VISIBLE_DEVICES=1 python ./mujoco_bear.py --seed=2 --dataset='hopper-medium-v0' --model='bear_hopper_m_params.json'&
#CUDA_VISIBLE_DEVICES=0 python ./mujoco_bc.py --seed=2 --dataset='hopper-medium-v0'&
#
## CUDA_VISIBLE_DEVICES=2 python ./mujoco_cql.py --seed=0 --dataset='walker2d-medium-v0' --model='cql_walk_m_params.json' &
## CUDA_VISIBLE_DEVICES=2 python ./mujoco_bcq.py --seed=0 --dataset='walker2d-medium-v0' --model='bcq_walk_m_params.json' &
## CUDA_VISIBLE_DEVICES=2 python ./mujoco_bear.py --seed=0 --dataset='walker2d-medium-v0' --model='bear_walk_m_params.json'&
## CUDA_VISIBLE_DEVICES=2 python ./mujoco_bc.py --seed=0 --dataset='walker2d-medium-v0'&
#
#
#CUDA_VISIBLE_DEVICES=1 python ./poisoned_mujoco_cql.py --seed=0 --dataset='walker2d-medium-v0' --model='cql_walk_m_params.json' &
#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_cql.py --seed=1 --dataset='walker2d-medium-v0' --model='cql_walk_m_params.json' &
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_cql.py --seed=2 --dataset='walker2d-medium-v0' --model='cql_walk_m_params.json' &

#CUDA_VISIBLE_DEVICES=3 python ./mujoco_bcq.py --seed=1 --dataset='walker2d-medium-v0' --model='bcq_walk_m_params.json' &
## CUDA_VISIBLE_DEVICES=3 python ./mujoco_bear.py --seed=1 --dataset='walker2d-medium-v0' --model='bear_walk_m_params.json'&
## CUDA_VISIBLE_DEVICES=3 python ./mujoco_bc.py --seed=1 --dataset='walker2d-medium-v0'&
#
#
#CUDA_VISIBLE_DEVICES=0 python ./mujoco_cql.py --seed=2 --dataset='walker2d-medium-v0' --model='cql_walk_m_params.json' &
## CUDA_VISIBLE_DEVICES=1 python ./mujoco_bcq.py --seed=2 --dataset='walker2d-medium-v0' --model='bcq_walk_m_params.json' &
## CUDA_VISIBLE_DEVICES=2 python ./mujoco_bear.py --seed=2 --dataset='walker2d-medium-v0' --model='bear_walk_m_params.json'&
#CUDA_VISIBLE_DEVICES=3 python ./mujoco_bc.py --seed=2 --dataset='walker2d-medium-v0'&

# ----------------------------------------------------------------------------------------------------------------------------------------------


#CUDA_VISIBLE_DEVICES=0 python ./mujoco_cql.py --seed=0 --dataset='ant-expert-v0' --model='cql_ant_m_params.json' &
#CUDA_VISIBLE_DEVICES=1 python ./mujoco_bcq.py --seed=0 --dataset='ant-expert-v0' --model='bcq_ant_m_params.json' &
#CUDA_VISIBLE_DEVICES=2 python ./mujoco_bear.py --seed=0 --dataset='ant-expert-v0' --model='bear_ant_m_params.json'&
#CUDA_VISIBLE_DEVICES=3 python ./mujoco_bc.py --seed=0 --dataset='ant-expert-v0'&

#CUDA_VISIBLE_DEVICES=0 python ./mujoco_cql.py --seed=1 --dataset='ant-expert-v0' --model='cql_ant_m_params.json' &
#CUDA_VISIBLE_DEVICES=1 python ./mujoco_bcq.py --seed=1 --dataset='ant-expert-v0' --model='bcq_ant_m_params.json' &
#CUDA_VISIBLE_DEVICES=2 python ./mujoco_bear.py --seed=1 --dataset='ant-expert-v0' --model='bear_ant_m_params.json'&
#CUDA_VISIBLE_DEVICES=3 python ./mujoco_bc.py --seed=1 --dataset='ant-expert-v0'&

#CUDA_VISIBLE_DEVICES=0 python ./mujoco_cql.py --seed=2 --dataset='ant-expert-v0' --model='cql_ant_m_params.json' &
#CUDA_VISIBLE_DEVICES=1 python ./mujoco_bcq.py --seed=2 --dataset='ant-expert-v0' --model='bcq_ant_m_params.json' &
#CUDA_VISIBLE_DEVICES=2 python ./mujoco_bear.py --seed=2 --dataset='ant-expert-v0' --model='bear_ant_m_params.json'&
#CUDA_VISIBLE_DEVICES=3 python ./mujoco_bc.py --seed=2 --dataset='ant-expert-v0'&

# --------------------------------------------------------------------------------------------------------------------------------------
# Poison training walker

#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_cql.py --seed=0 --dataset='walker2d-medium-v0' --model='cql_walk_m_params.json' --poison_rate=0.3  &
#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_bcq.py --seed=0 --dataset='walker2d-medium-v0' --model='bcq_walk_m_params.json' --poison_rate=0.3 &
#CUDA_VISIBLE_DEVICES=1 python ./poisoned_mujoco_bear.py --seed=0 --dataset='walker2d-medium-v0' --model='bear_walk_m_params.json' --poison_rate=0.3 &
#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_bc.py --seed=0 --dataset='walker2d-medium-v0' --poison_rate=0.3 &
#
#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_cql.py --seed=0 --dataset='walker2d-medium-v0' --model='cql_walk_m_params.json' --poison_rate=0.4  &
#CUDA_VISIBLE_DEVICES=1 python ./poisoned_mujoco_bcq.py --seed=0 --dataset='walker2d-medium-v0' --model='bcq_walk_m_params.json' --poison_rate=0.4 &
#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_bear.py --seed=0 --dataset='walker2d-medium-v0' --model='bear_walk_m_params.json' --poison_rate=0.4 &
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_bc.py --seed=0 --dataset='walker2d-medium-v0' --poison_rate=0.4 &

#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_cql.py --seed=0 --dataset='walker2d-medium-v0' --model='cql_walk_m_params.json' --poison_rate=0.3  &
#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_bcq.py --seed=0 --dataset='walker2d-medium-v0' --model='bcq_walk_m_params.json' --poison_rate=0.3 &
#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_bear.py --seed=0 --dataset='walker2d-medium-v0' --model='bear_walk_m_params.json' --poison_rate=0.3 &
#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_bc.py --seed=0 --dataset='walker2d-medium-v0' --poison_rate=0.3 &
#
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_cql.py --seed=0 --dataset='walker2d-medium-v0' --model='cql_walk_m_params.json' --poison_rate=0.4  &
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_bcq.py --seed=0 --dataset='walker2d-medium-v0' --model='bcq_walk_m_params.json' --poison_rate=0.4 &
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_bear.py --seed=0 --dataset='walker2d-medium-v0' --model='bear_walk_m_params.json' --poison_rate=0.4 &
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_bc.py --seed=0 --dataset='walker2d-medium-v0' --poison_rate=0.4 &

#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_cql.py --seed=1 --dataset='walker2d-medium-v0' --model='cql_walk_m_params.json' --poison_rate=0.2 &
#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_bcq.py --seed=1 --dataset='walker2d-medium-v0' --model='bcq_walk_m_params.json' --poison_rate=0.2 &
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_bear.py --seed=1 --dataset='walker2d-medium-v0' --model='bear_walk_m_params.json' --poison_rate=0.2 &
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_bc.py --seed=1 --dataset='walker2d-medium-v0' --poison_rate=0.2 &
#
#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_cql.py --seed=2 --dataset='walker2d-medium-v0' --model='cql_walk_m_params.json' --poison_rate=0.2 &
#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_bcq.py --seed=2 --dataset='walker2d-medium-v0' --model='bcq_walk_m_params.json' --poison_rate=0.2 &
#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_bear.py --seed=2 --dataset='walker2d-medium-v0' --model='bear_walk_m_params.json' --poison_rate=0.2 &
#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_bc.py --seed=2 --dataset='walker2d-medium-v0' --poison_rate=0.2 &

# --------------------------------------------------------------------------------------------------------------------------------------
# Poison training walker

#CUDA_VISIBLE_DEVICES=1 python ./poisoned_mujoco_cql.py --seed=0 --dataset='halfcheetah-medium-v0' --model='cql_half_m_params.json' --poison_rate=0.3 &
#CUDA_VISIBLE_DEVICES=1 python ./poisoned_mujoco_bcq.py --seed=0 --dataset='halfcheetah-medium-v0' --model='bcq_half_m_params.json' --poison_rate=0.3 &
#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_bear.py --seed=0 --dataset='halfcheetah-medium-v0' --model='bear_half_m_params.json' --poison_rate=0.3 &
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_bc.py --seed=0 --dataset='halfcheetah-medium-v0' --poison_rate=0.3 &
##
#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_cql.py --seed=1 --dataset='halfcheetah-medium-v0' --model='cql_half_m_params.json' --poison_rate=0.4 &
#CUDA_VISIBLE_DEVICES=2 python ./poisoned_mujoco_bcq.py --seed=1 --dataset='halfcheetah-medium-v0' --model='bcq_half_m_params.json' --poison_rate=0.4 &
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_bear.py --seed=1 --dataset='halfcheetah-medium-v0' --model='bear_half_m_params.json' --poison_rate=0.4 &
#CUDA_VISIBLE_DEVICES=3 python ./poisoned_mujoco_bc.py --seed=1 --dataset='halfcheetah-medium-v0' --poison_rate=0.4 &
##
#
## CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_cql.py --seed=2 --dataset='halfcheetah-medium-v0' --model='cql_half_m_params.json' --poison_rate=0.1 &
#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_bcq.py --seed=2 --dataset='halfcheetah-medium-v0' --model='bcq_half_m_params.json' --poison_rate=0.2 &
#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_bcq.py --seed=2 --dataset='halfcheetah-medium-v0' --model='bcq_half_m_params.json' --poison_rate=0.1 &
#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_bear.py --seed=2 --dataset='halfcheetah-medium-v0' --model='bear_half_m_params.json' --poison_rate=0.1 &
#CUDA_VISIBLE_DEVICES=0 python ./poisoned_mujoco_bc.py --seed=2 --dataset='halfcheetah-medium-v0' --poison_rate=0.1 &

