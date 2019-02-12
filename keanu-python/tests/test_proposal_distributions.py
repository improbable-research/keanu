import numpy as np
import pytest

from keanu import BayesNet, Model
from keanu.algorithm._proposal_distribution import ProposalDistribution
from keanu.vertex import Gaussian


@pytest.fixture
def net() -> BayesNet:
    with Model() as m:
        m.gamma = Gaussian(0., 1.)

    return m.to_bayes_net()


def test_you_can_create_a_prior_proposal_distribution(net) -> None:
    ProposalDistribution("prior", latents=net.get_latent_vertices())


def test_you_can_create_a_gaussian_proposal_distribution() -> None:
    ProposalDistribution("gaussian", sigma=np.array(1.))


def test_it_throws_if_you_specify_gaussian_without_a_value_for_sigma() -> None:
    with pytest.raises(TypeError, match=r"Gaussian Proposal Distribution requires a value for sigma"):
        ProposalDistribution("gaussian")


def test_it_throws_if_you_specify_sigma_but_the_type_isnt_gaussian() -> None:
    with pytest.raises(TypeError, match=r'Parameter sigma is not valid unless type is "gaussian"'):
        ProposalDistribution("prior", sigma=np.array(1.))


def test_it_throws_if_it_doesnt_recognise_the_type() -> None:
    with pytest.raises(KeyError, match=r"'foo'"):
        ProposalDistribution("foo")
