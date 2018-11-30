import numpy as np
import pytest

from keanu.algorithm._proposal_distribution import ProposalDistribution

def test_you_can_create_a_prior_proposal_distribution():
    ProposalDistribution("prior")

def test_you_can_create_a_gaussian_proposal_distribution():
    ProposalDistribution("gaussian", sigma = 1.)

def test_it_throws_if_it_doesnt_recognise_the_type():
    with pytest.raises(TypeError) as excinfo:
        ProposalDistribution("foo")

    assert str(excinfo.value == "Unknown Proposal Distribution type foo")



