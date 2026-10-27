import json

import pytest

from option_pricer.cli import main

ARGS = ["-S", "100", "-K", "100", "-T", "1", "-r", "0.05", "-v", "0.2"]


def test_prices_a_call_by_default(capsys):
    assert main(ARGS) == 0
    out = capsys.readouterr().out
    assert "Call" in out
    assert "10.450584" in out


def test_json_output_is_parseable(capsys):
    assert main([*ARGS, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"price", "delta", "gamma", "vega", "theta", "rho"}
    assert payload["price"] == pytest.approx(10.450583572, rel=1e-9)


def test_put_kind(capsys):
    assert main([*ARGS, "--kind", "put", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["price"] == pytest.approx(5.573526022, rel=1e-9)
    assert payload["delta"] < 0.0


def test_percent_and_decimal_inputs_agree(capsys):
    main([*ARGS, "--json"])
    decimal = json.loads(capsys.readouterr().out)
    main(["-S", "100", "-K", "100", "-T", "1", "-r", "5%", "-v", "20%", "--json"])
    percent = json.loads(capsys.readouterr().out)
    assert decimal == percent


def test_dividend_yield_lowers_call_value(capsys):
    main([*ARGS, "--json"])
    no_div = json.loads(capsys.readouterr().out)["price"]
    main([*ARGS, "-q", "3%", "--json"])
    with_div = json.loads(capsys.readouterr().out)["price"]
    assert with_div < no_div


def test_invalid_input_exits_with_code_2(capsys):
    assert main([*ARGS[:2], "-K", "-100", *ARGS[4:]]) == 2
    assert "error:" in capsys.readouterr().err


def test_missing_required_argument_errors():
    with pytest.raises(SystemExit) as exc:
        main(["-S", "100"])
    assert exc.value.code == 2
