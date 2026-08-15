import sys
import unittest
from pathlib import Path

import torch


TASK_DIR = (
    Path(__file__).resolve().parents[2]
    / "isaaclab_k1_locomotion"
    / "tasks"
    / "manager_based"
    / "walk_kick_likelihood"
)
sys.path.insert(0, str(TASK_DIR))

from agents.model import DirectKickingActorCritic  # noqa: E402


EXPECTED_STATE_SHAPES = {
    "logstd": (1, 12),
    "actor.encoder.lstm.weight_ih_l0": (256, 6),
    "actor.encoder.lstm.weight_hh_l0": (256, 64),
    "actor.encoder.lstm.bias_ih_l0": (256,),
    "actor.encoder.lstm.bias_hh_l0": (256,),
    "actor.network.0.weight": (256, 118),
    "actor.network.0.bias": (256,),
    "actor.network.2.weight": (128, 256),
    "actor.network.2.bias": (128,),
    "actor.network.4.weight": (128, 128),
    "actor.network.4.bias": (128,),
    "actor.network.6.weight": (12, 128),
    "actor.network.6.bias": (12,),
    "critic.encoder.lstm.weight_ih_l0": (256, 6),
    "critic.encoder.lstm.weight_hh_l0": (256, 64),
    "critic.encoder.lstm.bias_ih_l0": (256,),
    "critic.encoder.lstm.bias_hh_l0": (256,),
    "critic.network.0.weight": (256, 138),
    "critic.network.0.bias": (256,),
    "critic.network.2.weight": (256, 256),
    "critic.network.2.bias": (256,),
    "critic.network.4.weight": (128, 256),
    "critic.network.4.bias": (128,),
    "critic.network.6.weight": (1, 128),
    "critic.network.6.bias": (1,),
}


def make_observations(batch_size=4):
    return {
        "policy": torch.randn(batch_size, 132),
        "critic": torch.randn(batch_size, 20),
    }


def make_model(observations=None):
    if observations is None:
        observations = make_observations()
    return DirectKickingActorCritic(
        observations,
        {"policy": ["policy"], "critic": ["critic"]},
        12,
    )


class DirectKickingActorCriticTest(unittest.TestCase):
    def test_state_dict_has_exact_source_keys_and_shapes(self):
        model = make_model()

        state = model.state_dict()

        self.assertEqual(set(state), set(EXPECTED_STATE_SHAPES))
        self.assertEqual(len(state), 25)
        for key, shape in EXPECTED_STATE_SHAPES.items():
            self.assertEqual(tuple(state[key].shape), shape, key)
        torch.testing.assert_close(model.logstd, torch.full((1, 12), -2.0))

    def test_actor_critic_shapes_match_rsl_rl_api(self):
        observations = make_observations(batch_size=6)
        model = make_model(observations)

        actions = model.act(observations)
        values = model.evaluate(observations)
        log_probability = model.get_actions_log_prob(actions)

        self.assertEqual(tuple(actions.shape), (6, 12))
        self.assertEqual(tuple(values.shape), (6, 1))
        self.assertEqual(tuple(log_probability.shape), (6,))
        self.assertEqual(tuple(model.action_mean.shape), (6, 12))
        self.assertEqual(tuple(model.action_std.shape), (6, 12))
        self.assertEqual(tuple(model.entropy.shape), (6,))

    def test_horizon_lstm_is_stateless_across_policy_calls(self):
        observations = make_observations(batch_size=3)
        model = make_model(observations)

        first = model.act_inference(observations)
        model.act_inference(make_observations(batch_size=3))
        second = model.act_inference(observations)

        torch.testing.assert_close(second, first, rtol=0.0, atol=0.0)
        self.assertFalse(model.is_recurrent)
        self.assertFalse(hasattr(model, "get_hidden_states"))

    def test_log_std_alias_preserves_checkpoint_key_and_reset_path(self):
        model = make_model()

        self.assertEqual(model.noise_std_type, "log")
        self.assertEqual(model.log_std.data_ptr(), model.logstd.data_ptr())
        with torch.no_grad():
            model.log_std.data.clamp_(min=-1.0)

        torch.testing.assert_close(model.logstd, torch.full((1, 12), -1.0))
        self.assertIn("logstd", model.state_dict())
        self.assertNotIn("log_std", model.state_dict())

    def test_critic_group_may_include_policy_prefix(self):
        observations = make_observations(batch_size=2)
        model = DirectKickingActorCritic(
            observations,
            {"policy": ["policy"], "critic": ["policy", "critic"]},
            12,
        )

        self.assertEqual(tuple(model.evaluate(observations).shape), (2, 1))
        self.assertEqual(tuple(model.get_critic_obs(observations).shape), (2, 152))

    def test_actor_is_torchscript_exportable(self):
        model = make_model().eval()
        observation = torch.randn(2, 132)

        scripted_actor = torch.jit.script(model.actor)

        torch.testing.assert_close(scripted_actor(observation), model.actor(observation))


if __name__ == "__main__":
    unittest.main()
