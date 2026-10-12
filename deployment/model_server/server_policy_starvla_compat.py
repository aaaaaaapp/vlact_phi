# StarVLA RobotWin eval client compatibility layer.
#
# Starvla/examples/Robotwin/eval_files/model2robotwin_interface.py expects websocket
# metadata fields such as action_chunk_size and response["data"]["actions"].
# The default server_policy.py keeps the original minimal metadata for other
# benchmarks (LIBERO, SimplerEnv, etc.).
#
# Use this entry only when pairing VLAct checkpoints with Starvla RobotWin eval.sh.

import logging
import os
import socket

import torch
from accelerate import PartialState

from deployment.model_server.server_policy import build_argparser, start_debugpy_once
from deployment.model_server.starvla_robotwin_serve_wrapper import (
    StarvlaRobotwinServeWrapper,
)
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework import build_framework
from starVLA.model.framework.share_tools import (
    dict_to_namespace,
    read_mode_config,
)
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


JOINT_EXTENSION_PREFIXES = (
    "pi_discrete_conditioner",
    "pi_action_layer_gates",
    "pi_physics_layer_gates",
    "action_token_probe",
    "action_condition_embedding",
    "physics_token_probe",
)


def load_policy_model(args):
    """Load either a standard full checkpoint or a strict joint extension."""
    model_config, norm_stats = read_mode_config(args.ckpt_path)
    if str((model_config.get("framework") or {}).get("name")) != "QwenPIv4JointTokens":
        return baseframework.from_pretrained(args.ckpt_path)

    joint = model_config["framework"]["joint_token_prediction"]
    if args.joint_condition_source is not None:
        joint["condition_source"] = args.joint_condition_source
    if args.joint_soft_temperature is not None:
        if args.joint_soft_temperature <= 0.0:
            raise ValueError("--joint-soft-temperature must be positive")
        joint["soft_temperature"] = args.joint_soft_temperature

    config = dict_to_namespace(model_config)
    model = build_framework(cfg=config)
    base_checkpoint = str(config.trainer.pretrained_checkpoint)
    model = TrainerUtils.load_extension_checkpoint(
        model,
        base_checkpoint,
        JOINT_EXTENSION_PREFIXES,
    )
    head_state = torch.load(
        args.ckpt_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    required_keys = {
        key
        for key in model.state_dict()
        if key.startswith(tuple(f"{prefix}." for prefix in JOINT_EXTENSION_PREFIXES))
    }
    if set(head_state) != required_keys:
        raise RuntimeError(
            "joint deployment checkpoint must exactly contain extension tensors: "
            f"missing={sorted(required_keys - set(head_state))[:8]}, "
            f"unexpected={sorted(set(head_state) - required_keys)[:8]}"
        )
    incompatible = model.load_state_dict(head_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_extension = [
        key for key in incompatible.missing_keys if key in required_keys
    ]
    if unexpected or missing_extension:
        raise RuntimeError(
            "failed to load joint deployment extension: "
            f"missing={missing_extension[:8]}, unexpected={unexpected[:8]}"
        )
    model.norm_stats = norm_stats
    model.set_discrete_conditioning_enabled(
        not args.disable_discrete_conditioning
    )
    model.set_physics_condition_enabled(
        not args.disable_physics_conditioning
    )
    return model


def main(args) -> None:
    PartialState()

    vla = load_policy_model(args)
    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()

    if args.selector_threshold is not None:
        from deployment.model_server.conservative_episode_selector import ConservativeEpisodeSelector
        vla = ConservativeEpisodeSelector(vla, args.selector_threshold,
            args.selector_radius, args.selector_trace)

    policy = StarvlaRobotwinServeWrapper(
        framework=vla,
        ckpt_path=args.ckpt_path,
        default_unnorm_key=os.environ.get("ROBOTWIN_UNNORM_KEY"),
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating StarVLA-compat server (host: %s, ip: %s)", hostname, local_ip)

    metadata = policy.metadata
    if args.action_chunk_size is not None:
        metadata = dict(metadata)
        metadata["action_chunk_size"] = int(args.action_chunk_size)
    logging.info("Server metadata: %s", metadata)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    )
    logging.info("server running ...")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    parser.add_argument(
        "--joint-condition-source",
        choices=("cached", "predicted", "soft"),
        default=None,
    )
    parser.add_argument("--joint-soft-temperature", type=float, default=None)
    parser.add_argument("--disable-discrete-conditioning", action="store_true")
    parser.add_argument("--disable-physics-conditioning", action="store_true")
    parser.add_argument("--selector-threshold", type=float, default=None)
    parser.add_argument("--selector-radius", type=float, default=0.2)
    parser.add_argument("--selector-trace", default="selector_trace.jsonl")
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
