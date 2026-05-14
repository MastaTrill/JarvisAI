import pytest
from fastapi.testclient import TestClient
try:
    from main_api import app as main_app
except Exception:
    main_app = None

@pytest.mark.skipif(not main_app, reason="main_app not available")
def test_self_healing_api_trigger_and_events():
    client = TestClient(main_app)
    # Trigger a self-healing event (should recover from negative input)
    resp = client.post("/system/self-healing/trigger?x=-1")
    assert resp.status_code == 200
    data = resp.json()
    assert "event" in data
    assert data["event"]["status"] in ("recovered", "failed")
    # Fetch the event log
    resp2 = client.get("/system/self-healing/events")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert "events" in data2
    assert isinstance(data2["events"], list)
import pytest
from advanced_features.self_healing import SelfHealingAI

def always_fails():
    raise ValueError("Simulated failure")

def succeeds_on_second_try():
    if not hasattr(succeeds_on_second_try, "called"):
        succeeds_on_second_try.called = True
        raise RuntimeError("First call fails")
    return "success"

def test_self_healing_run_recovers():
    sh_ai = SelfHealingAI(max_retries=2)
    # Reset state for idempotency
    if hasattr(succeeds_on_second_try, "called"):
        del succeeds_on_second_try.called
    result = sh_ai.run(succeeds_on_second_try)
    assert result == "success"

def test_self_healing_run_fails():
    sh_ai = SelfHealingAI(max_retries=1)
    result = sh_ai.run(always_fails)
    assert result is None

def test_self_heal_decorator():
    sh_ai = SelfHealingAI(max_retries=1)
    @sh_ai.self_heal_decorator
    def sometimes_fails():
        if not hasattr(sometimes_fails, "called"):
            sometimes_fails.called = True
            raise Exception("fail once")
        return 42
    # Reset state for idempotency
    if hasattr(sometimes_fails, "called"):
        del sometimes_fails.called
    assert sometimes_fails() == 42
import unittest
from advanced_features.self_healing import SelfHealingPipeline

class TestSelfHealingPipeline(unittest.TestCase):
    def test_monitor_fix(self):
        shp = SelfHealingPipeline()
        pipeline = {'status': 'error'}
        result = shp.monitor(pipeline)
        self.assertTrue(result['fixed'])
        self.assertEqual(result['status'], 'recovered')

    def test_monitor_ok(self):
        shp = SelfHealingPipeline()
        pipeline = {'status': 'ok'}
        result = shp.monitor(pipeline)
        self.assertNotIn('fixed', result)
        self.assertEqual(result['status'], 'ok')

if __name__ == "__main__":
    unittest.main()