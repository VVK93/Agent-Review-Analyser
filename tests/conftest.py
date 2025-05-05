import pytest
from data_types import Persona, FeatureProposal

@pytest.fixture(scope="session")
def sample_personas():
    """Shared fixture for sample personas used across multiple tests"""
    return [
        Persona(
            name="User1",
            background="Music lover",
            quote="I love music",
            sentiment="positive",
            pain_points=["No dark mode"]
        ),
        Persona(
            name="User2",
            background="Podcast listener",
            quote="Podcasts are great",
            sentiment="neutral",
            pain_points=["Too many ads"]
        )
    ]

@pytest.fixture(scope="session")
def sample_features():
    """Shared fixture for sample feature proposals used across multiple tests"""
    return [
        FeatureProposal(id=1, description="Add dark mode"),
        FeatureProposal(id=2, description="Remove ads")
    ]

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Automatically mock environment variables for all tests"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key") 