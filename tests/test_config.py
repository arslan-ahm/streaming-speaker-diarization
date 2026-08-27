"""Config seams: type coercion, `_base_` inheritance, override precedence, strictness.

These are the tests that stop a silent ablation failure. ``--set
diarizer.spawn_enabled=false`` producing the *string* ``"false"`` — which is
truthy — would make the no-spawn ablation identical to the baseline and the
results table would show a mechanism contributing nothing when in fact it was
never disabled.
"""

from __future__ import annotations

import pytest

from streamdiar.config import (
    Config,
    DiarizerConfig,
    _coerce,
    apply_overrides,
    deep_merge,
    from_dict,
    load_config,
    load_yaml_tree,
    parse_set_override,
    save_config,
)


class TestCoercion:
    def test_bool_false_strings_become_false(self):
        for text in ["false", "False", "FALSE", "0", "no", "off"]:
            assert _coerce(text, bool) is False, text

    def test_bool_true_strings_become_true(self):
        for text in ["true", "True", "1", "yes", "on"]:
            assert _coerce(text, bool) is True, text

    def test_bool_rejects_nonsense_rather_than_guessing(self):
        with pytest.raises(ValueError, match="Cannot read"):
            _coerce("maybe", bool)

    def test_int_accepts_scientific_notation(self):
        assert _coerce("1e3", int) == 1000

    def test_int_rejects_a_real_truncation(self):
        with pytest.raises(ValueError, match="not an integer"):
            _coerce("1.5", int)

    def test_int_accepts_float_valued_integer(self):
        assert _coerce(3.0, int) == 3

    def test_float_from_string(self):
        assert _coerce("1e-3", float) == pytest.approx(1e-3)

    def test_tuple_from_comma_string(self):
        assert _coerce("1,2,4", tuple[int, ...]) == (1, 2, 4)

    def test_tuple_from_bracketed_string(self):
        assert _coerce("[1, 2, 4]", tuple[int, ...]) == (1, 2, 4)

    def test_tuple_of_floats(self):
        assert _coerce("0.1,0.9", tuple[float, float]) == (0.1, 0.9)

    def test_optional_none_strings(self):
        for text in ["none", "None", "null", ""]:
            assert _coerce(text, str | None) is None

    def test_optional_passes_through_a_value(self):
        assert _coerce("data/ami", str | None) == "data/ami"

    def test_empty_tuple_string_is_empty(self):
        assert _coerce("", tuple[int, ...]) == ()


class TestFromDict:
    def test_unknown_top_level_key_raises(self):
        with pytest.raises(KeyError, match="Unknown config key"):
            from_dict({"nope": 1}, Config)

    def test_unknown_nested_key_raises(self):
        with pytest.raises(KeyError, match="DiarizerConfig"):
            from_dict({"diarizer": {"laetncy_budget_ms": 1}}, Config)

    def test_error_message_lists_known_keys(self):
        with pytest.raises(KeyError, match="spawn_threshold"):
            from_dict({"diarizer": {"xyz": 1}}, Config)

    def test_nested_dataclasses_are_constructed(self):
        cfg = from_dict({"diarizer": {"max_speakers": 3}}, Config)
        assert isinstance(cfg.diarizer, DiarizerConfig)
        assert cfg.diarizer.max_speakers == 3

    def test_absent_keys_keep_defaults(self):
        cfg = from_dict({}, Config)
        assert cfg.diarizer.method == "online"

    def test_empty_nested_mapping_is_allowed(self):
        cfg = from_dict({"diarizer": None}, Config)
        assert cfg.diarizer.method == "online"

    def test_non_dataclass_target_raises(self):
        with pytest.raises(TypeError, match="not a dataclass"):
            from_dict({}, dict)


class TestDeepMerge:
    def test_override_wins(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge_keeps_unmentioned_keys(self):
        out = deep_merge({"d": {"a": 1, "b": 2}}, {"d": {"b": 3}})
        assert out == {"d": {"a": 1, "b": 3}}

    def test_inputs_are_not_mutated(self):
        base = {"d": {"a": 1}}
        deep_merge(base, {"d": {"a": 2}})
        assert base == {"d": {"a": 1}}

    def test_scalar_replaces_dict(self):
        assert deep_merge({"d": {"a": 1}}, {"d": 5}) == {"d": 5}


class TestOverrides:
    def test_parse_splits_on_first_equals(self):
        keys, value = parse_set_override("a.b=c=d")
        assert keys == ["a", "b"]
        assert value == "c=d"

    def test_parse_requires_equals(self):
        with pytest.raises(ValueError, match="key=value"):
            parse_set_override("a.b")

    def test_parse_requires_nonempty_key(self):
        with pytest.raises(ValueError, match="non-empty key"):
            parse_set_override("=5")

    def test_apply_creates_missing_nesting(self):
        out = apply_overrides({}, ["diarizer.max_speakers=4"])
        assert out == {"diarizer": {"max_speakers": "4"}}

    def test_apply_does_not_mutate_input(self):
        payload = {"diarizer": {"max_speakers": 2}}
        apply_overrides(payload, ["diarizer.max_speakers=9"])
        assert payload["diarizer"]["max_speakers"] == 2

    def test_bool_override_is_a_real_bool(self):
        cfg = load_config(None, ["diarizer.spawn_enabled=false"])
        assert cfg.diarizer.spawn_enabled is False

    def test_bool_override_true_is_a_real_bool(self):
        cfg = load_config(None, ["embedder.causal=true"])
        assert cfg.embedder.causal is True

    def test_float_override_is_typed_by_the_field(self):
        cfg = load_config(None, ["train.lr=1e-3"])
        assert isinstance(cfg.train.lr, float)
        assert cfg.train.lr == pytest.approx(1e-3)

    def test_tuple_override(self):
        cfg = load_config(None, ["embedder.dilations=1,2,4"])
        assert cfg.embedder.dilations == (1, 2, 4)


class TestYamlInheritance:
    def test_base_is_resolved(self, tmp_path):
        (tmp_path / "a.yaml").write_text("seed: 7\ndiarizer:\n  max_speakers: 3\n")
        (tmp_path / "b.yaml").write_text("_base_: a.yaml\nname: child\n")
        payload = load_yaml_tree(tmp_path / "b.yaml")
        assert payload["seed"] == 7
        assert payload["name"] == "child"
        assert payload["diarizer"]["max_speakers"] == 3

    def test_child_overrides_base(self, tmp_path):
        (tmp_path / "a.yaml").write_text("seed: 7\n")
        (tmp_path / "b.yaml").write_text("_base_: a.yaml\nseed: 9\n")
        assert load_yaml_tree(tmp_path / "b.yaml")["seed"] == 9

    def test_multiple_bases_apply_in_order(self, tmp_path):
        (tmp_path / "a.yaml").write_text("seed: 1\nname: a\n")
        (tmp_path / "b.yaml").write_text("seed: 2\n")
        (tmp_path / "c.yaml").write_text("_base_: [a.yaml, b.yaml]\n")
        payload = load_yaml_tree(tmp_path / "c.yaml")
        assert payload["seed"] == 2  # later base wins
        assert payload["name"] == "a"

    def test_nested_base_chain(self, tmp_path):
        (tmp_path / "a.yaml").write_text("seed: 1\n")
        (tmp_path / "b.yaml").write_text("_base_: a.yaml\nname: b\n")
        (tmp_path / "c.yaml").write_text("_base_: b.yaml\nout_root: r\n")
        payload = load_yaml_tree(tmp_path / "c.yaml")
        assert (payload["seed"], payload["name"], payload["out_root"]) == (1, "b", "r")

    def test_cycle_raises_with_the_chain(self, tmp_path):
        (tmp_path / "a.yaml").write_text("_base_: b.yaml\n")
        (tmp_path / "b.yaml").write_text("_base_: a.yaml\n")
        with pytest.raises(ValueError, match="Cyclic"):
            load_yaml_tree(tmp_path / "a.yaml")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_yaml_tree(tmp_path / "nope.yaml")

    def test_non_mapping_top_level_raises(self, tmp_path):
        (tmp_path / "a.yaml").write_text("- 1\n- 2\n")
        with pytest.raises(TypeError, match="mapping at top level"):
            load_yaml_tree(tmp_path / "a.yaml")

    def test_empty_yaml_is_an_empty_mapping(self, tmp_path):
        (tmp_path / "a.yaml").write_text("")
        assert load_yaml_tree(tmp_path / "a.yaml") == {}

    def test_base_is_relative_to_the_including_file(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.yaml").write_text("seed: 5\n")
        (sub / "b.yaml").write_text("_base_: a.yaml\n")
        assert load_yaml_tree(sub / "b.yaml")["seed"] == 5


class TestPrecedence:
    def test_override_precedence_chain(self, tmp_path):
        """defaults < _base_ < named file < --set < kwargs, in that order."""
        (tmp_path / "a.yaml").write_text("seed: 1\n")
        (tmp_path / "b.yaml").write_text("_base_: a.yaml\nseed: 2\n")
        # base file only
        assert load_config(tmp_path / "a.yaml").seed == 1
        # child file beats its base
        assert load_config(tmp_path / "b.yaml").seed == 2
        # --set beats the file
        assert load_config(tmp_path / "b.yaml", ["seed=3"]).seed == 3
        # kwargs beat --set
        assert load_config(tmp_path / "b.yaml", ["seed=3"], seed=4).seed == 4

    def test_defaults_apply_with_no_file(self):
        assert load_config(None).seed == 0

    def test_later_set_wins_over_earlier(self):
        cfg = load_config(None, ["seed=1", "seed=2"])
        assert cfg.seed == 2


class TestDerivedQuantities:
    def test_frames_per_window(self):
        cfg = load_config(None, ["diarizer.window_ms=1000", "data.frame_rate=100"])
        assert cfg.frames_per_window == 100

    def test_frames_per_hop(self):
        cfg = load_config(None, ["diarizer.hop_ms=250", "data.frame_rate=100"])
        assert cfg.frames_per_hop == 25

    def test_frames_per_window_is_at_least_one(self):
        cfg = load_config(None, ["diarizer.window_ms=1", "data.frame_rate=100"])
        assert cfg.frames_per_window == 1

    def test_budget_windows_floors(self):
        """A budget shorter than one hop buys no lookahead: the next window does not exist."""
        cfg = load_config(None, ["diarizer.hop_ms=250", "diarizer.latency_budget_ms=125"])
        assert cfg.budget_windows == 0

    def test_budget_windows_exact_multiple(self):
        cfg = load_config(None, ["diarizer.hop_ms=250", "diarizer.latency_budget_ms=1000"])
        assert cfg.budget_windows == 4

    def test_budget_windows_rounds_down_not_nearest(self):
        cfg = load_config(None, ["diarizer.hop_ms=250", "diarizer.latency_budget_ms=749"])
        assert cfg.budget_windows == 2

    def test_zero_budget_is_zero_windows(self):
        cfg = load_config(None, ["diarizer.latency_budget_ms=0"])
        assert cfg.budget_windows == 0


class TestRoundTrip:
    def test_to_dict_is_nested(self):
        d = load_config(None).to_dict()
        assert d["diarizer"]["method"] == "online"

    def test_tuples_become_lists_for_yaml(self):
        d = load_config(None).to_dict()
        assert isinstance(d["embedder"]["dilations"], list)

    def test_yaml_round_trip_preserves_every_field(self, tmp_path):
        cfg = load_config(None, ["diarizer.spawn_enabled=false", "train.lr=1e-4"])
        path = save_config(cfg, tmp_path / "c.yaml")
        again = load_config(path)
        assert again.to_dict() == cfg.to_dict()

    def test_saved_config_uses_lf_endings(self, tmp_path):
        path = save_config(load_config(None), tmp_path / "c.yaml")
        assert b"\r\n" not in path.read_bytes()

    def test_shipped_configs_all_load(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "configs"
        names = sorted(p.name for p in root.glob("*.yaml"))
        assert len(names) >= 8
        for name in names:
            cfg = load_config(root / name)
            assert cfg.name

    def test_smoke_config_is_actually_small(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "configs"
        cfg = load_config(root / "smoke.yaml")
        assert cfg.train.steps <= 50
        assert cfg.data.n_test_recordings <= 5

    def test_ablation_configs_each_flip_exactly_one_switch(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "configs"
        base = load_config(root / "base.yaml")
        for name, field in [
            ("ablation_no_bounded_window.yaml", ("diarizer", "bounded_window")),
            ("ablation_no_spawn.yaml", ("diarizer", "spawn_enabled")),
            ("ablation_noncausal.yaml", ("embedder", "causal")),
        ]:
            cfg = load_config(root / name)
            diffs = _differing_fields(base, cfg)
            # `name` always differs; the switch is the only other difference.
            assert diffs == {"name", ".".join(field)}, (name, diffs)


def _differing_fields(a, b) -> set[str]:
    """Dotted names of every scalar field that differs between two configs."""
    out: set[str] = set()

    def walk(x, y, prefix: str) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for k in set(x) | set(y):
                walk(x.get(k), y.get(k), f"{prefix}{k}." if False else (f"{prefix}.{k}" if prefix else k))
        elif x != y:
            out.add(prefix)

    walk(a.to_dict(), b.to_dict(), "")
    return out
