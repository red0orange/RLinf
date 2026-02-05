"""
Register a PickCube variant that uses the custom AlohaMini SO100 V2 robot.

Why this file exists:
- ManiSkill's builtin `PickCube-v1` only allows a fixed set of robot_uids.
- RLinf imports all modules under `rlinf.envs.maniskill.tasks` on startup, so
  adding this file is enough to register a new env id in a pure-incremental way.

This env keeps ManiSkill task logic unchanged and only:
- ensures the AlohaMini agent is registered (by importing the module that calls
  `@register_agent`)
- extends SUPPORTED_ROBOTS to include `aloha_mini_so100_v2`
- reuses the SO100 task config (camera/workspace tuning) for AlohaMini-SO100
"""

from __future__ import annotations

from typing import Any

try:
    # Importing this module registers the agent via @register_agent.
    import AlohaMini.maniskill.agents.aloha_mini.aloha_mini_so100_v2  # noqa: F401
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import the custom AlohaMini ManiSkill agent. "
        "Make sure the RLinf repo root is on PYTHONPATH so that "
        "`AlohaMini/maniskill/...` is importable."
    ) from e

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube import PICK_CUBE_CONFIGS, PickCubeEnv
from mani_skill.utils.registration import register_env
import torch


ALOHA_MINI_SO100_UID = "aloha_mini_so100_v2"


@register_env("PickCubeAlohaMiniSO100LeftArm-v1", max_episode_steps=50)
class PickCubeAlohaMiniSO100LeftArmEnv(PickCubeEnv):
    """PickCube task with custom AlohaMini-SO100 robot."""

    SUPPORTED_ROBOTS = list(PickCubeEnv.SUPPORTED_ROBOTS) + [ALOHA_MINI_SO100_UID]

    def __init__(
        self,
        *args,
        robot_uids: str = ALOHA_MINI_SO100_UID,
        robot_init_qpos_noise: float = 0.02,
        cube_half_size_scale: float = 2.0,
        # Reward knobs (keep ManiSkill PickCube structure but tuned for AlohaMini).
        reward_reach_tanh_scale: float = 2.0,
        reward_place_tanh_scale: float = 5.0,
        reward_static_tanh_scale: float = 5.0,
        reward_success_value: float = 5.0,
        **kwargs,
    ):
        # Keep PickCubeEnv behavior, but reuse the "so100" task config for AlohaMini-SO100.
        self.robot_init_qpos_noise = robot_init_qpos_noise

        if robot_uids == ALOHA_MINI_SO100_UID:
            cfg = PICK_CUBE_CONFIGS["so100"]
        elif robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]

        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]

        if cube_half_size_scale != 1.0:
            s = float(cube_half_size_scale)
            if isinstance(self.cube_half_size, (list, tuple)):
                self.cube_half_size = [float(x) * s for x in self.cube_half_size]
            else:
                self.cube_half_size = float(self.cube_half_size) * s

        # Store reward parameters for this custom robot/env id.
        self._reward_reach_tanh_scale = float(reward_reach_tanh_scale)
        self._reward_place_tanh_scale = float(reward_place_tanh_scale)
        self._reward_static_tanh_scale = float(reward_static_tanh_scale)
        self._reward_success_value = float(reward_success_value)

        # Call BaseEnv.__init__ directly to avoid PickCubeEnv.__init__ overriding cfg selection.
        BaseEnv.__init__(self, *args, robot_uids=robot_uids, **kwargs)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """
        Dense reward for PickCube with AlohaMini robot.

        This follows ManiSkill's builtin PickCube dense reward structure, but fixes
        the robot-specific static term for AlohaMini (mobile base + 2 grippers):
        - exclude base joints (to match `AlohaMiniBaseAgent.is_static()` behavior)
        - exclude both gripper joints so gripper chatter doesn't dominate static reward

        Additionally, `reward_reach_tanh_scale` is lowered by default so reaching
        reward is still informative when the custom robot placement starts far away.
        """
        # reaching
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, dim=1
        )
        reaching_reward = 1.0 - torch.tanh(
            self._reward_reach_tanh_scale * tcp_to_obj_dist
        )
        reward = reaching_reward

        # grasp
        is_grasped = info["is_grasped"].to(dtype=reward.dtype)
        reward = reward + is_grasped

        # place (only matters if grasped)
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, dim=1
        )
        place_reward = 1.0 - torch.tanh(
            self._reward_place_tanh_scale * obj_to_goal_dist
        )
        reward = reward + place_reward * is_grasped

        # static (only matters if object placed)
        qvel = self.agent.robot.get_qvel()
        if self.robot_uids == ALOHA_MINI_SO100_UID:
            # AlohaMiniSO100V2 joint order (16 DOF):
            # [base(3), lift(1), left_arm(6 incl gripper), right_arm(6 incl gripper)]
            # Align with agent.is_static(): exclude base joints.
            qvel = qvel[:, 3:]
            # Exclude both grippers (left gripper, right gripper).
            # After excluding base: 13 dims -> [lift(1), left(6), right(6)]
            if qvel.shape[1] == 13:
                keep = torch.ones(13, dtype=torch.bool, device=qvel.device)
                keep[6] = False   # left gripper
                keep[12] = False  # right gripper
                qvel = qvel[:, keep]
        else:
            # Keep ManiSkill's original behavior for builtin robots.
            if self.robot_uids in ["panda", "widowxai"]:
                qvel = qvel[..., :-2]
            elif self.robot_uids == "so100":
                qvel = qvel[..., :-1]

        static_reward = 1.0 - torch.tanh(
            self._reward_static_tanh_scale * torch.linalg.norm(qvel, dim=1)
        )
        is_obj_placed = info["is_obj_placed"].to(dtype=reward.dtype)
        reward = reward + static_reward * is_obj_placed

        # success override (keep ManiSkill convention)
        success = info["success"]
        if torch.is_tensor(success):
            reward[success] = self._reward_success_value
        else:
            success_t = torch.as_tensor(
                success, device=reward.device, dtype=torch.bool
            )
            reward[success_t] = self._reward_success_value
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        # Match ManiSkill's normalized_dense convention.
        denom = self._reward_success_value if self._reward_success_value != 0 else 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / denom

    def _load_agent(self, options: dict):  # @note 机器人本体初始化位置
        """
        Override PickCubeEnv robot spawn pose.

        ManiSkill's builtin PickCube uses an initial pose tailored for Panda.
        For AlohaMini-SO100 we reuse the SO100 placement (x offset + 90deg yaw),
        which keeps the robot reachable to the cube workspace on the table.
        """
        import numpy as np
        import sapien
        from transforms3d.euler import euler2quat

        if self.robot_uids == ALOHA_MINI_SO100_UID:
            # Match ManiSkill's SO100 placement (see TableSceneBuilder.initialize).
            super(PickCubeEnv, self)._load_agent(
                options,
                sapien.Pose(p=[-0.7, 0.0, -0.7], q=euler2quat(0, 0, np.pi / 2)),
            )
        else:
            super()._load_agent(options)

    def _initialize_episode(self, env_idx, options: dict):
        # Let PickCube initialize the table, cube, goal, etc.
        super()._initialize_episode(env_idx, options)

        # If using AlohaMini, we must also set a sensible initial qpos because
        # ManiSkill's TableSceneBuilder.initialize() does not know this robot uid.
        if self.robot_uids != ALOHA_MINI_SO100_UID:
            return

        import numpy as np
        import torch
        import sapien
        from transforms3d.euler import euler2quat

        with torch.device(self.device):
            b = len(env_idx)
            # Prefer the agent's keyframe as a stable starting pose.
            # Users can tune this keyframe in the agent implementation.
            qpos0 = np.array(self.agent.keyframes.get("ready", self.agent.keyframes["rest"]).qpos, dtype=np.float32)
            qpos = np.repeat(qpos0[None, :], b, axis=0)
            self.agent.reset(qpos)

            # Also enforce a consistent base pose each episode.
            self.agent.robot.set_pose(
                sapien.Pose(p=[-0.7, 0.0, -0.7], q=euler2quat(0, 0, np.pi / 2))
            )

            # cam_main is configured in the agent (`AlohaMiniSO100V2._sensor_configs`).
            # We keep it fixed per-task for stability across vectorized envs.


if __name__ == "__main__":  # pragma: no cover
    """
    Headless-friendly camera debug helper.

    Usage examples:
      - Save one cam_main image:
        python rlinf/envs/maniskill/tasks/pick_cube_aloha_mini_so100_left_arm.py --out cam_main.png

      - Change resolution:
        python rlinf/envs/maniskill/tasks/pick_cube_aloha_mini_so100_left_arm.py --width 640 --height 480
    """

    import argparse
    from pathlib import Path

    import gymnasium as gym
    import numpy as np
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="cam_main.png", help="Output image path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fov", type=float, default=None, help="Override cam_main fov (radians). Optional.")
    parser.add_argument(
        "--control-mode",
        type=str,
        default="pd_joint_delta_pos",
        choices=["pd_joint_delta_pos", "pd_joint_pos"],
        help="ManiSkill control_mode to debug. Use pd_joint_delta_pos to test [-1,1] -> delta mapping.",
    )
    parser.add_argument(
        "--debug-steps",
        type=int,
        default=30,
        help="How many consecutive env.step() to apply per debug action before saving image.",
    )
    parser.add_argument(
        "--sim-backend",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="ManiSkill simulation backend",
    )
    args = parser.parse_args()

    # Create env (raw ManiSkill env, not RLinf wrapper).
    # `obs_mode='rgb'` is required for sensor_data images.
    env = gym.make(
        "PickCubeAlohaMiniSO100LeftArm-v1",
        obs_mode="rgb",
        render_mode="rgb_array",
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        sensor_configs=dict(width=args.width, height=args.height),
    )

    obs, info = env.reset(seed=args.seed)
    cam = obs["sensor_data"]["cam_main"]["rgb"]

    # cam can be [H, W, 3] or [B, H, W, 3] depending on backend/env.
    cam_np = np.asarray(cam.cpu())
    if cam_np.ndim == 4:
        cam_np = cam_np[0]

    # Optional: override fov at runtime for quick debug (best-effort).
    # Many ManiSkill versions build sensors at init; this won't always apply.
    if args.fov is not None:
        try:
            env.unwrapped._sensor_configs["cam_main"].fov = float(args.fov)
        except Exception:
            pass

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save image. Prefer imageio; fall back to PIL if needed.
    def _imwrite(path: Path, img: np.ndarray):
        """Best-effort image writer for headless debug."""
        try:
            import imageio.v2 as imageio  # type: ignore

            imageio.imwrite(path.as_posix(), img)
            return
        except Exception:
            pass
        try:
            from PIL import Image  # type: ignore

            Image.fromarray(img).save(path.as_posix())
            return
        except Exception:
            pass
        raise RuntimeError("Failed to save image (need imageio or PIL installed).")

    _imwrite(out_path, cam_np)

    print(f"[ok] saved cam_main to: {out_path.resolve()}")

    # ---------------------------------------------------------------------
    # Joint-control sanity check (headless friendly)
    #
    # Goal: verify the meaning/scale of [-1, 1] actions for AlohaMini.
    #
    # For AlohaMiniSO100V2 under `control_mode="pd_joint_delta_pos"`, ManiSkill uses
    # PDJointPosController with:
    #   - use_delta=True
    #   - normalize_action=True (default)
    # So an action in [-1, 1] is mapped to the per-joint delta bounds (lower/upper)
    # configured in the agent (e.g. +/-0.05 rad for arm joints).
    # ---------------------------------------------------------------------
    try:
        print("\n===== Joint-control debug (AlohaMini) =====")
        print("env_id:", "PickCubeAlohaMiniSO100LeftArm-v1")
        print("control_mode:", getattr(env.unwrapped, "control_mode", None))
        if hasattr(env, "action_space"):
            asp = env.action_space
            try:
                print("action_space:", asp)
                if hasattr(asp, "low") and hasattr(asp, "high"):
                    print("action_space.low[0:10]:", np.array(asp.low).reshape(-1)[:10])
                    print("action_space.high[0:10]:", np.array(asp.high).reshape(-1)[:10])
            except Exception:
                pass

        agent = env.unwrapped.agent
        try:
            print("left_arm_joint_names:", getattr(agent, "left_arm_joint_names", None))
        except Exception:
            pass
        ctrl = getattr(agent, "controller", None)
        left_ctrl = None
        if ctrl is not None and hasattr(ctrl, "controllers") and isinstance(ctrl.controllers, dict):
            left_ctrl = ctrl.controllers.get("left_arm", None)

        if left_ctrl is None:
            print("[warn] Could not locate agent.controller.controllers['left_arm']; skip detailed controller debug.")
        else:
            cfg = left_ctrl.config
            print("left_arm controller:", type(left_ctrl))
            print("  normalize_action:", getattr(cfg, "normalize_action", None))
            print("  use_delta:", getattr(cfg, "use_delta", None))
            print("  lower:", getattr(cfg, "lower", None))
            print("  upper:", getattr(cfg, "upper", None))

            # Some control modes (e.g. pd_joint_pos) may not have lower/upper.
            _lower = getattr(cfg, "lower", None)
            _upper = getattr(cfg, "upper", None)
            lower = torch.tensor(_lower if _lower is not None else [0.0] * 6, dtype=torch.float32)
            upper = torch.tensor(_upper if _upper is not None else [0.0] * 6, dtype=torch.float32)

            def _map_norm_to_delta(a_norm: torch.Tensor) -> torch.Tensor:
                """
                Map normalized action in [-1, 1] to delta in [lower, upper].
                This matches ManiSkill controller behavior when normalize_action=True.
                """
                # a_norm in [-1, 1] -> t in [0, 1]
                t = (a_norm + 1.0) * 0.5
                return lower + t * (upper - lower)

            def _safe_tag(s: str) -> str:
                # Keep filenames simple across shells/filesystems.
                return (
                    s.replace("+", "plus")
                    .replace("-", "minus")
                    .replace("=", "_")
                    .replace(" ", "_")
                    .replace("/", "_")
                )

            def _step_and_report(action16: np.ndarray, tag: str, do_reset: bool = False):
                if do_reset:
                    # Reset to the same initial state so comparisons are meaningful.
                    # Note: this also re-randomizes cube unless the env is configured otherwise.
                    env.reset(seed=args.seed)
                a16 = action16.astype(np.float32, copy=False)
                qpos0 = agent.robot.get_qpos().detach().cpu().numpy().reshape(-1)
                last_obs = None
                last_rew = None
                last_term = None
                last_trunc = None
                last_info = None

                # Step multiple times so PD has time to track the target.
                for _ in range(max(1, int(args.debug_steps))):
                    last_obs, last_rew, last_term, last_trunc, last_info = env.step(a16)
                    # Stop early if episode ended
                    done = False
                    try:
                        if torch.is_tensor(last_term):
                            done = bool(last_term.any().item())
                        else:
                            done = bool(last_term)
                        if torch.is_tensor(last_trunc):
                            done = done or bool(last_trunc.any().item())
                        else:
                            done = done or bool(last_trunc)
                    except Exception:
                        done = False
                    if done:
                        break

                qpos1 = agent.robot.get_qpos().detach().cpu().numpy().reshape(-1)

                # left arm is [4:10] in the 16D combined action/qpos layout
                dq_left = qpos1[4:10] - qpos0[4:10]

                # what delta was *commanded* by controller mapping (best-effort)
                a_left_norm = torch.tensor(a16[4:10], dtype=torch.float32).clamp(-1, 1)
                if getattr(cfg, "normalize_action", False) and _lower is not None and _upper is not None:
                    a_left_cmd = _map_norm_to_delta(a_left_norm)
                    cmd_label = "mapped_delta (rad)"
                else:
                    # For pd_joint_pos (absolute), this is just the commanded joint targets slice.
                    a_left_cmd = a_left_norm
                    cmd_label = "commanded_left (raw)"

                tgt = None
                try:
                    if hasattr(left_ctrl, "_target_qpos") and left_ctrl._target_qpos is not None:
                        tgt = left_ctrl._target_qpos.detach().cpu().numpy().reshape(-1)
                except Exception:
                    tgt = None

                rew_val = last_rew
                if torch.is_tensor(rew_val):
                    rew_val = float(rew_val.mean().item())
                print(f"\n[{tag}] steps={max(1, int(args.debug_steps))} rew={rew_val}")
                print("  action16.left[4:10] (norm):", np.round(a16[4:10], 3))
                print(f"  {cmd_label}:", np.round(a_left_cmd.cpu().numpy(), 4))
                print("  actual dq_left (rad):", np.round(dq_left, 4))
                if tgt is not None:
                    print("  controller _target_qpos (6d):", np.round(tgt, 4))

                # Save an image for this debug step (same directory as --out).
                try:
                    if last_obs is None:
                        raise RuntimeError("no obs collected")
                    cam_t = last_obs["sensor_data"]["cam_main"]["rgb"]
                    cam_np2 = np.asarray(cam_t.detach().cpu()) if torch.is_tensor(cam_t) else np.asarray(cam_t)
                    if cam_np2.ndim == 4:
                        cam_np2 = cam_np2[0]
                    img_path = out_path.with_name(f"{out_path.stem}_{_safe_tag(tag)}{out_path.suffix}")
                    _imwrite(img_path, cam_np2)
                    print("  saved cam_main:", img_path.resolve())
                except Exception as e:
                    print("  [warn] failed to save cam_main for", tag, ":", repr(e))

            # Baseline: zero action (should result in tiny motion)
            a = np.zeros(env.action_space.shape, dtype=np.float32)
            _step_and_report(a, "zero")

            # Single-joint impulses (normalized +1 / -1) on left arm joint 0 (index 4)
            a = np.zeros(env.action_space.shape, dtype=np.float32)
            a[4] = 1.0
            _step_and_report(a, "left_j0=+1")

            a = np.zeros(env.action_space.shape, dtype=np.float32)
            a[4] = -1.0
            _step_and_report(a, "left_j0=-1")

            # All left-arm joints = +1 (max delta within bounds)
            a = np.zeros(env.action_space.shape, dtype=np.float32)
            a[4:10] = 1.0
            _step_and_report(a, "left_all=+1")

            # ---------------------------------------------------------
            # Per-joint min/max sweep (reset before every snapshot)
            # ---------------------------------------------------------
            print("\n===== Per-joint sweep (left arm) =====")
            # Indices for left arm within the 16D action: [4:10]
            for j in range(6):
                a = np.zeros(env.action_space.shape, dtype=np.float32)
                a[4 + j] = -1.0
                _step_and_report(a, f"left_joint{j}_min(-1)", do_reset=True)

                a = np.zeros(env.action_space.shape, dtype=np.float32)
                a[4 + j] = 1.0
                _step_and_report(a, f"left_joint{j}_max(+1)", do_reset=True)
            print("\n===== Per-joint sweep done =====")

            # Show RLinf padding behavior for 6D -> 16D (best-effort).
            try:
                from rlinf.envs.action_utils import prepare_actions_for_maniskill

                raw6 = torch.tensor([[1, -1, 0.5, -0.5, 0.0, 1.0]], dtype=torch.float32)  # [B,6]
                padded = prepare_actions_for_maniskill(
                    raw_chunk_actions=raw6, num_action_chunks=1, action_dim=6, action_scale=1.0, policy="aloha_left_arm_joint_dpos"
                )
                if torch.is_tensor(padded):
                    p0 = padded[0].detach().cpu().numpy()
                    print("\n[rlinf padding] raw6:", raw6[0].cpu().numpy())
                    print("[rlinf padding] padded16[0:12]:", np.round(p0[:12], 3))
                    print("[rlinf padding] left slice padded16[4:10]:", np.round(p0[4:10], 3))
            except Exception as e:
                print("[warn] skip rlinf padding check:", repr(e))

        print("\n===== Joint-control debug done =====\n")
    except Exception as e:
        print("[warn] joint-control debug failed:", repr(e))

    env.close()
