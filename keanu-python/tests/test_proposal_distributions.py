import numpy as np
import pytest

from keanu import BayesNet, Model
from keanu.algorithm._proposal_distribution import ProposalDistribution
from keanu.vartypes import tensor_arg_types
from keanu.vertex import Gamma, Gaussian


@pytest.fixture
def net() -> BayesNet:
    with Model() as m:
        m.gamma = Gamma(1., 1.)
        m.gaussian = Gaussian(0., m.gamma)

    return m.to_bayes_net()


def test_you_can_create_a_prior_proposal_distribution(net) -> None:
    ProposalDistribution("prior", latents=list(net.iter_latent_vertices()))


@pytest.mark.parametrize("sigma", [[1., 2.], [np.array(1.), np.array(2.)]])
def test_you_can_create_a_gaussian_proposal_distribution(sigma: tensor_arg_types, net: BayesNet) -> None:
    ProposalDistribution("gaussian", latents=list(net.iter_latent_vertices()), sigma=sigma)


def test_it_defaults_if_you_specify_gaussian_without_a_value_for_sigma(net: BayesNet) -> None:
    ProposalDistribution("gaussian")


def test_it_throws_if_you_specify_gaussian_with_not_enough_sigmas_for_each_latent(net: BayesNet) -> None:
    with pytest.raises(
            TypeError, match="Gaussian Proposal Distribution requires a list of sigmas. One for each latent."):
        ProposalDistribution("gaussian", latents=list(net.iter_latent_vertices()), sigma=[1.])


def test_it_allows_using_default_sigma_if_you_specify_gaussian_without_values_for_latents() -> None:
    ProposalDistribution("gaussian", default_sigma=1.5)


def test_it_throws_if_you_specify_gaussian_with_empty_list_of_latents(net: BayesNet) -> None:
    with pytest.raises(
            TypeError, match="Gaussian Proposal Distribution requires a list of sigmas. One for each latent."):
        ProposalDistribution("gaussian", latents=[], sigma=[])


def test_it_throws_if_you_specify_sigma_but_the_type_isnt_gaussian() -> None:
    with pytest.raises(TypeError, match=r'Parameter sigma is not valid unless type is "gaussian"'):
        ProposalDistribution("prior", sigma=1.)


def test_it_throws_if_it_doesnt_recognise_the_type() -> None:
    with pytest.raises(KeyError, match=r"'foo'"):
        ProposalDistribution("foo")
