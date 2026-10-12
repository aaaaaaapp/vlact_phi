"""Experimental episode selector; decisions never use seed or success labels."""
import json
import numpy as np
import torch

class ConservativeEpisodeSelector:
    def __init__(self, model, threshold, radius, trace_path):
        if threshold <= 0 or radius <= 0:
            raise ValueError("threshold/radius must be positive")
        self.model, self.threshold, self.radius = model, threshold, radius
        self.trace = open(trace_path, "a", buffering=1)
        self.enabled, self.step = None, 0

    def reset_memory(self):
        self.enabled, self.step = None, 0

    def predict_action(self, examples=None, **kwargs):
        cpu_rng = torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        original = self.model.discrete_conditioning_enabled
        try:
            self.model.set_discrete_conditioning_enabled(False)
            for key in ("_pi_residual_token_count", "_pi_action_residual_token_count", "_pi_physics_residual_token_count"):
                setattr(self.model, key, 0)
            base = self.model.predict_action(examples=examples, **kwargs)
            torch.set_rng_state(cpu_rng)
            if gpu_rng:
                torch.cuda.set_rng_state_all(gpu_rng)
            self.model.set_discrete_conditioning_enabled(True)
            joint = self.model.predict_action(examples=examples, **kwargs)
        finally:
            self.model.set_discrete_conditioning_enabled(original)
        a = np.asarray(base["normalized_actions"], dtype=np.float32)
        b = np.asarray(joint["normalized_actions"], dtype=np.float32)
        if a.shape[0] != 1:
            raise ValueError("one episode per server required")
        delta = b - a
        rms = float(np.sqrt(np.mean(delta[..., :12] ** 2)))
        if self.enabled is None:
            self.enabled = rms <= self.threshold
        active = self.enabled and rms <= 2 * self.threshold
        norm = np.linalg.norm(delta[..., :12], axis=-1, keepdims=True)
        scale = np.minimum(1.0, self.radius / np.maximum(norm, 1e-8))
        out = a.copy()
        if active:
            out[..., :12] += delta[..., :12] * scale
            out[..., 12:] = b[..., 12:]
        self.trace.write(json.dumps({"step": self.step, "rms": rms,
            "episode_enabled": bool(self.enabled), "active": bool(active),
            "min_scale": float(scale.min()), "base": a.tolist(),
            "tanh": b.tolist(), "selected": out.tolist()}) + "\n")
        self.step += 1
        return {"normalized_actions": out}
