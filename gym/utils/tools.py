import torch
from isaacgym.torch_utils import quat_rotate, quat_conjugate, quat_mul
from utils.gym_info import control_ik_damping
def quat_axis(q, axis=0):
    '''
    :func apply rotation represented by quanternion `q`
    on basis vector(along axis)
    :return vector after rotation
    '''
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def control_ik(j_eef, device, dpose, num_envs):

    # Set controller parameters
    # IK params
    damping = control_ik_damping
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