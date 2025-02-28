import torch

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.actuators import ImplicitActuator
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter


from omni.isaac.lab.envs import ManagerBasedEnv

import os
from omni.isaac.leggedloco.config.h1.motion_utils.motion_lib import MotionLib


def reset_camera_pos_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_delta_range: dict[str, tuple[float, float]],
):
    """Reset the camera to a random position uniformly within the given ranges.

    This function randomizes the position of the asset.

    * It samples the delta position from the given ranges and adds them to the default camera position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    camera: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    pos_w, quat_w = camera._compute_view_world_poses(env_ids)

    # poses
    range_list = [pose_delta_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=camera.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=camera.device)

    positions = pos_w + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(quat_w, orientations_delta)

    # set into the physics simulation
    camera.set_world_poses(positions, orientations, env_ids=env_ids, convention="world")


def init_motion(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        device, motion_name="01_01.npy",
        no_keybody=True,
        regen_pkl=False
):
    # ['pelvis', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 'left_hip_roll_link',
    # 'right_hip_roll_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_hip_pitch_link',
    # t_hip_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_knee_link',
    # 'right_knee_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_ankle_link',
    # 'right_ankle_link', 'left_elbow_link', 'right_elbow_link']

    # ['left_hip_yaw', 'right_hip_yaw', 'torso', 'left_hip_roll', 'right_hip_roll',
    # 'left_shoulder_pitch', 'right_shoulder_pitch', 'left_hip_pitch', 'right_hip_pitch',
    # 'left_shoulder_roll', 'right_shoulder_roll', 'left_knee', 'right_knee',
    # 'left_shoulder_yaw', 'right_shoulder_yaw', 'left_ankle', 'right_ankle', 'left_elbow', 'right_elbow']
    no_keybody = no_keybody
    regen_pkl = regen_pkl
    _key_body_ids = torch.tensor([3, 6, 9, 12], device=device)  # self._build_key_body_ids_tensor(key_bodies)
    # ['pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link',
    # 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link',
    # 'torso_link',
    # 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_hand_keypoint_link',
    # 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_hand_keypoint_link']
    _key_body_ids_sim = torch.tensor([1, 4, 5,  # Left Hip yaw, Knee, Ankle
                                           6, 9, 10,
                                           12, 15, 16,  # Left Shoulder pitch, Elbow, hand
                                           17, 20, 21], device=device)
    _key_body_ids_sim_subset = torch.tensor([6, 7, 8, 9, 10, 11], device=device)  # no knee and ankle

    _num_key_bodies = len(_key_body_ids_sim_subset)
    _dof_body_ids = [1, 2, 3,  # Hip, Knee, Ankle
                          4, 5, 6,
                          7,  # Torso
                          8, 9, 10,  # Shoulder, Elbow, Hand
                          11, 12, 13]  # 13
    _dof_offsets = [0, 3, 4, 5, 8, 9, 10,
                         11,
                         14, 15, 16, 19, 20, 21]  # 14
    _valid_dof_body_ids = torch.ones(len(_dof_body_ids) + 2 * 4, device=device, dtype=torch.bool)
    _valid_dof_body_ids[-1] = 0
    _valid_dof_body_ids[-6] = 0
    dof_indices_sim = torch.tensor([0, 1, 2, 5, 6, 7, 11, 12, 13, 16, 17, 18], device=device,
                                        dtype=torch.long)
    dof_indices_motion = torch.tensor([2, 0, 1, 7, 5, 6, 12, 11, 13, 17, 16, 18], device=device,
                                           dtype=torch.long)

    # self._dof_ids_subset = torch.tensor([0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device)  # no knee and ankle
    _dof_ids_subset = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18], device=device)  # no knee and ankle
    _n_demo_dof = len(_dof_ids_subset)

    # ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
    # 'left_knee_joint', 'left_ankle_joint',
    # 'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
    # 'right_knee_joint', 'right_ankle_joint',
    # 'torso_joint',
    # 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
    # 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint']
    # self.dof_ids_subset = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device, dtype=torch.long)
    # motion_name = "17_04_stealth"

    # if cfg.motion.motion_type == "single":
    #     motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/retarget_npy/{cfg.motion.motion_name}.npy")
    # else:
    #     assert cfg.motion.motion_type == "yaml"
    #     motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/configs/{cfg.motion.motion_name}")

    motion_file = os.path.join(f"C:\\Users\\Administrator\\Desktop\\{motion_name}")

    _motion_lib = MotionLib(motion_file=motion_file,
                                dof_body_ids=_dof_body_ids,
                                dof_offsets=_dof_offsets,
                                key_body_ids=_key_body_ids.cpu().numpy(),
                                device=device,
                                no_keybody=no_keybody,
                                regen_pkl=regen_pkl)

    env.motion_lab = _motion_lib