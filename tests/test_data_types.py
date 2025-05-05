import pytest
from data_types import FeatureProposal, Persona

def test_feature_proposal_creation():
    """Test basic FeatureProposal creation and markdown formatting"""
    feature = FeatureProposal(id=1, description="Add dark mode")
    assert feature.id == 1
    assert feature.description == "Add dark mode"
    assert feature.md() == "1. Add dark mode"

def test_persona_creation():
    """Test basic Persona creation with all required fields"""
    persona = Persona(
        name="John Doe",
        background="Music enthusiast",
        quote="Music is my life",
        sentiment="positive",
        pain_points=["Limited playlist customization", "No offline mode"]
    )
    assert persona.name == "John Doe"
    assert persona.background == "Music enthusiast"
    assert persona.quote == "Music is my life"
    assert persona.sentiment == "positive"
    assert len(persona.pain_points) == 2
    assert "Limited playlist customization" in persona.pain_points

def test_persona_system_prompt():
    """Test system prompt generation for Persona"""
    persona = Persona(
        name="Jane Smith",
        background="Professional DJ",
        quote="Music is my passion",
        sentiment="neutral",
        pain_points=["Limited DJ tools", "No crossfade control"]
    )
    prompt = persona.system_prompt
    assert "You are Jane Smith" in prompt
    assert "Professional DJ" in prompt
    assert "neutral" in prompt
    assert "Limited DJ tools" in prompt
    assert "crossfade control" in prompt

def test_persona_markdown():
    """Test markdown representation of Persona"""
    persona = Persona(
        name="Alex Brown",
        background="Student",
        quote="I love discovering new music",
        sentiment="positive",
        pain_points=["Ads are annoying", "Battery drain"]
    )
    md = persona.md()
    assert "### Alex Brown" in md
    assert "*I love discovering new music*" in md
    assert "**Background**: Student" in md
    assert "**Sentiment**: Positive" in md
    assert "**Key Pain Points**" in md
    assert "- Ads are annoying" in md
    assert "- Battery drain" in md

def test_persona_optional_fields():
    """Test Persona creation with optional fields"""
    persona = Persona(
        name="Test User",
        background="Test background",
        quote="Test quote",
        sentiment="neutral",
        pain_points=[],
        inspired_by_cluster_id="cluster_123"
    )
    assert persona.inspired_by_cluster_id == "cluster_123" 