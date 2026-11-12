import socket

import pytest

from src.api import agent_api


def _fake_resolver(mapping):
    def _resolve(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, port)) for addr in mapping[host]]
    return _resolve


@pytest.mark.parametrize(
    "addr",
    ["127.0.0.1", "10.0.0.5", "192.168.1.10", "172.16.0.1", "169.254.169.254", "0.0.0.0", "::1", "fd00::1", "::ffff:127.0.0.1"],
)
def test_rejects_non_public_targets(monkeypatch, addr):
    monkeypatch.delenv("JARVIS_ALLOW_PRIVATE_WEBHOOKS", raising=False)
    monkeypatch.setattr(agent_api.socket, "getaddrinfo", _fake_resolver({"hook.test": [addr]}))
    with pytest.raises(ValueError):
        agent_api._assert_public_webhook_url("https://hook.test/x")


def test_rejects_if_any_resolved_address_is_private(monkeypatch):
    monkeypatch.delenv("JARVIS_ALLOW_PRIVATE_WEBHOOKS", raising=False)
    monkeypatch.setattr(agent_api.socket, "getaddrinfo", _fake_resolver({"hook.test": ["93.184.216.34", "10.0.0.1"]}))
    with pytest.raises(ValueError):
        agent_api._assert_public_webhook_url("https://hook.test/x")


def test_rejects_non_http_scheme():
    with pytest.raises(ValueError):
        agent_api._assert_public_webhook_url("file:///etc/passwd")


def test_allows_public_target(monkeypatch):
    monkeypatch.delenv("JARVIS_ALLOW_PRIVATE_WEBHOOKS", raising=False)
    monkeypatch.setattr(agent_api.socket, "getaddrinfo", _fake_resolver({"hook.test": ["93.184.216.34"]}))
    agent_api._assert_public_webhook_url("https://hook.test/x")


def test_dev_override_allows_private(monkeypatch):
    monkeypatch.setenv("JARVIS_ALLOW_PRIVATE_WEBHOOKS", "1")
    agent_api._assert_public_webhook_url("http://127.0.0.1:9000/hook")


def test_dispatch_refuses_private_without_network_call(monkeypatch):
    monkeypatch.delenv("JARVIS_ALLOW_PRIVATE_WEBHOOKS", raising=False)
    monkeypatch.setattr(agent_api.socket, "getaddrinfo", _fake_resolver({"internal.test": ["10.1.2.3"]}))
    calls = []
    monkeypatch.setattr(agent_api, "_urlopen_http", lambda *a, **k: calls.append(a))
    result = agent_api._dispatch_webhook_notification({"enabled": True, "webhook_url": "https://internal.test/h"}, {"m": 1})
    assert result["sent"] is False and "non-public" in result["error"]
    assert calls == []
