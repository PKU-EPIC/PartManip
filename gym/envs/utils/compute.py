import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
    quat_mul, tensor_clamp

def deal_with_dof_state(state_tensor, cabinet_target_id, cabinet_dof_num_list, franka_dof_num, env_num, env_per_asset):
    env_dof_sate_list =[]
    new_tensor = torch.zeros((env_num, franka_dof_num + 1, state_tensor.shape[-1]), device = state_tensor.device)
    mask = torch.zeros((env_num, franka_dof_num + 1), device = state_tensor.device)
    now = 0

    assert(state_tensor.shape[0] == env_num*franka_dof_num + torch.tensor(cabinet_dof_num_list).sum()*env_per_asset)
    for i_ in range(env_num):
        mask[i_,0:franka_dof_num] = torch.arange(now, now+franka_dof_num, device = state_tensor.device)
        mask[i_,franka_dof_num] = now+franka_dof_num+cabinet_target_id[i_//env_per_asset]
        new_tensor[i_,0:franka_dof_num] = state_tensor[now:now+franka_dof_num]
        new_tensor[i_, franka_dof_num] = state_tensor[now+franka_dof_num+cabinet_target_id[i_//env_per_asset]]
        env_dof_sate_list.append(state_tensor[now: now + franka_dof_num + cabinet_dof_num_list[i_//env_per_asset]])
        now += franka_dof_num + cabinet_dof_num_list[i_//env_per_asset]
        

    eval_tensor = state_tensor[mask.long()]
    mask = mask.long()

    assert((eval_tensor == new_tensor).any())
    return new_tensor, mask, env_dof_sate_list

def deal_with_rigid_state(state_tensor, part_ids, handle_ids, base_ids,  cabinet_rigid_num_list, franka_rigid_num, env_num, env_per_asset):
    new_tensor = torch.zeros((env_num, franka_rigid_num + 3, state_tensor.shape[-1]), device = state_tensor.device)
    mask = torch.zeros((env_num, franka_rigid_num + 3), device = state_tensor.device)
    now = 0
    assert(state_tensor.shape[0] == env_num* franka_rigid_num + torch.tensor(cabinet_rigid_num_list).sum()*env_per_asset)
    for i_ in range(env_num):
        mask[i_,0: franka_rigid_num] = torch.arange(now, now+franka_rigid_num, device = state_tensor.device)
        mask[i_, franka_rigid_num] = now+franka_rigid_num+base_ids[i_//env_per_asset]
        mask[i_, franka_rigid_num+1] = now+franka_rigid_num+part_ids[i_//env_per_asset]
        mask[i_, franka_rigid_num+2] = now+franka_rigid_num+handle_ids[i_//env_per_asset]
        new_tensor[i_,0: franka_rigid_num] = state_tensor[now:now+franka_rigid_num]
        new_tensor[i_, franka_rigid_num] =   state_tensor[now+franka_rigid_num+base_ids[i_//env_per_asset]]
        new_tensor[i_, franka_rigid_num+1] = state_tensor[now+franka_rigid_num+part_ids[i_//env_per_asset]]
        new_tensor[i_, franka_rigid_num+2] = state_tensor[now+franka_rigid_num+handle_ids[i_//env_per_asset]]
        now += franka_rigid_num + cabinet_rigid_num_list[i_//env_per_asset]

    eval_tensor = state_tensor[mask.long()]
    mask = mask.long()

    assert((eval_tensor == new_tensor).any())
    return new_tensor, mask

def deal_with_rigid_state_nohandle(state_tensor, part_ids,  base_ids,  cabinet_rigid_num_list, franka_rigid_num, env_num, env_per_asset):
    new_tensor = torch.zeros((env_num, franka_rigid_num + 3, state_tensor.shape[-1]), device = state_tensor.device)
    mask = torch.zeros((env_num, franka_rigid_num + 3), device = state_tensor.device)
    now = 0
    assert(state_tensor.shape[0] == env_num* franka_rigid_num + torch.tensor(cabinet_rigid_num_list).sum()*env_per_asset)
    for i_ in range(env_num):
        mask[i_,0: franka_rigid_num] = torch.arange(now, now+franka_rigid_num, device = state_tensor.device)
        mask[i_, franka_rigid_num] = now+franka_rigid_num+base_ids[i_//env_per_asset]
        mask[i_, franka_rigid_num+1] = now+franka_rigid_num+part_ids[i_//env_per_asset]
        # mask[i_, franka_rigid_num+2] = now+franka_rigid_num+handle_ids[i_//env_per_asset]
        new_tensor[i_,0: franka_rigid_num] = state_tensor[now:now+franka_rigid_num]
        new_tensor[i_, franka_rigid_num] =   state_tensor[now+franka_rigid_num+base_ids[i_//env_per_asset]]
        new_tensor[i_, franka_rigid_num+1] = state_tensor[now+franka_rigid_num+part_ids[i_//env_per_asset]]
        # new_tensor[i_, franka_rigid_num+2] = state_tensor[now+franka_rigid_num+handle_ids[i_//env_per_asset]]
        now += franka_rigid_num + cabinet_rigid_num_list[i_//env_per_asset]

    eval_tensor = state_tensor[mask.long()]
    mask = mask.long()

    assert((eval_tensor == new_tensor).any())
    return new_tensor, mask



def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def control_ik(j_eef, device, dpose, num_envs):

    # Set controller parameters
    # IK params
    damping = 0.05
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u

def relative_pose(src, dst) :

    shape = dst.shape
    p = dst.view(-1, shape[-1])[:, :3] - src.view(-1, src.shape[-1])[:, :3]
    ip = dst.view(-1, shape[-1])[:, 3:]
    ret = torch.cat((p, ip), dim=1)
    return ret.view(*shape)

def quat_axis(q, axis=0):
    '''
    :func apply rotation represented by quanternion `q`
    on basis vector(along axis)
    :return vector after rotation
    '''
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)
  