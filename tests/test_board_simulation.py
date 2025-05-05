import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from data_types import Persona, FeatureProposal
from board_simulation import (
    get_random_llm,
    generate_persona_agents,
    simulate_userboard
)

@pytest.fixture
def sample_personas():
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

@pytest.fixture
def sample_features():
    return [
        FeatureProposal(id=1, description="Add dark mode"),
        FeatureProposal(id=2, description="Remove ads")
    ]

def test_get_random_llm():
    """Test that get_random_llm returns a ChatOpenAI instance with valid parameters"""
    llm = get_random_llm()
    assert llm.model_name in ["gpt-4o", "gpt-4.1", "gpt-4.1-mini"]
    assert 0.3 <= llm.temperature <= 1.0

@pytest.mark.asyncio
async def test_generate_persona_agents(sample_personas):
    """Test that persona agents are generated correctly"""
    with patch('board_simulation.get_random_llm') as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4.1"
        mock_llm.temperature = 0.5
        mock_get_llm.return_value = mock_llm

        agents = generate_persona_agents(sample_personas)
        
        assert len(agents) == len(sample_personas)
        for agent, persona in zip(agents, sample_personas):
            assert agent.name == persona.name
            assert agent.model == "gpt-4.1"
            assert agent.model_settings.temperature == 0.5
            assert persona.name in agent.instructions
            assert persona.background in agent.instructions
            assert persona.sentiment in agent.instructions
            assert all(pain_point in agent.instructions for pain_point in persona.pain_points)

@pytest.mark.asyncio
async def test_simulate_userboard_empty_inputs():
    """Test that simulate_userboard handles empty inputs gracefully"""
    transcript, history = await simulate_userboard([], [])
    assert transcript == ""
    assert history == []

@pytest.mark.asyncio
async def test_simulate_userboard_basic(sample_personas, sample_features):
    """Test basic board simulation functionality"""
    with patch('board_simulation.Runner.run') as mock_run:
        # Mock responses for facilitator and agents
        mock_run.side_effect = [
            AsyncMock(final_output="What are your thoughts?"),
            AsyncMock(final_output="I like dark mode"),
            AsyncMock(final_output="I prefer no ads"),
            AsyncMock(final_output="No follow-up needed")
        ]

        transcript, history = await simulate_userboard(sample_personas, sample_features, rounds=1)
        
        assert transcript != ""
        assert len(history) > 0
        assert "What are your thoughts?" in transcript
        assert "I like dark mode" in transcript
        assert "I prefer no ads" in transcript

@pytest.mark.asyncio
async def test_simulate_userboard_followup(sample_personas, sample_features):
    """Test that follow-up questions are handled correctly"""
    with patch('board_simulation.Runner.run') as mock_run:
        # Mock responses including a follow-up question
        mock_run.side_effect = [
            AsyncMock(final_output="Initial question"),
            AsyncMock(final_output="First response"),
            AsyncMock(final_output="Follow-up question"),
            AsyncMock(final_output="Follow-up response"),
            AsyncMock(final_output="No follow-up needed")
        ]

        transcript, history = await simulate_userboard(sample_personas, sample_features, rounds=1)
        
        assert "Initial question" in transcript
        assert "First response" in transcript
        assert "Follow-up question" in transcript
        assert "Follow-up response" in transcript 