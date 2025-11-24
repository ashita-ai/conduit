import pytest
from conduit.engines.router import Router
from conduit.core.config import settings

def test_window_size_propagation(mocker):
    """Test that window_size setting is correctly propagated to LinUCBBandit."""
    # Mock settings to have a specific window size
    mocker.patch.object(settings, "bandit_window_size", 500)
    
    # Initialize router
    router = Router()
    
    # Check if window_size reached the bandit
    assert router.hybrid_router.linucb.window_size == 500
    
    # Verify default behavior (mocking 0)
    mocker.patch.object(settings, "bandit_window_size", 0)
    router_default = Router()
    assert router_default.hybrid_router.linucb.window_size == 0
