from __future__ import annotations

from unittest.mock import patch

from obvs.lenses import TokenIdentity


class TestTokenIdentity:
    @staticmethod
    def test_run_compute_outputs():
        ti1 = TokenIdentity("The quick brown fox jumps over the lazy", device="cpu")

        source_layers = range(ti1._patchscope.n_layers_source // 2)
        ti1.run(source_layers)

        ti2 = TokenIdentity("The quick brown fox jumps over the lazy", device="cpu")

        source_layers = range(ti2._patchscope.n_layers_source // 2)
        # Mock the nextloop method so it just saves the outputs
        with patch("obvs.lenses.TokenIdentity._nextloop") as mock_nextloop:
            mock_nextloop.return_value = None
            ti2.run_and_compute(source_layers)
        assert ti1.source_layers == ti2.source_layers

        assert len(mock_nextloop.call_args_list) == len(ti1.outputs)

    @staticmethod
    def test_run_compute_surprisals():
        ti1 = TokenIdentity("The quick brown fox jumps over the lazy", device="cpu")

        source_layers = range(ti1._patchscope.n_layers_source // 2)
        ti1.run(source_layers).compute_surprisal()

        ti2 = TokenIdentity("The quick brown fox jumps over the lazy", device="cpu")

        source_layers = range(ti2._patchscope.n_layers_source // 2)
        # Mock the nextloop method so it just saves the outputs
        ti2.run_and_compute(source_layers)
        assert (ti1.surprisal == ti2.surprisal).all()
