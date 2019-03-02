import numpy as np
import pytest

from keanu import BayesNet, Model
from keanu.algorithm._proposal_distribution import ProposalDistribution
from keanu.vertex import Gamma, Gaussian


@pytest.fixture
def net() -> BayesNet:
    with Model() as m:
        m.gamma = Gamma(1., 1.)
        m.gaussian = Gaussian(0., m.gamma)

    return m.to_bayes_net()


def test_you_can_create_a_prior_proposal_distribution(net) -> None:
    ProposalDistribution("prior", latents=list(net.iter_latent_vertices()))


def test_you_can_create_a_gaussian_proposal_distribution() -> None:
    ProposalDistribution("gaussian", sigma=1.)


def test_you_can_create_a_multivariate_gaussian_proposal_distribution(net: BayesNet) -> None:
    ProposalDistribution("multivariate_gaussian", latents=list(net.iter_latent_vertices()), sigma=[1., 2.])


def test_it_throws_if_you_specify_gaussian_without_a_value_for_sigma() -> None:
    with pytest.raises(TypeError, match=r"Gaussian Proposal Distribution requires a value for sigma"):
        ProposalDistribution("gaussian")


def test_it_throws_if_you_specify_gaussian_with_sigma_that_is_not_tensor_arg_type() -> None:
    with pytest.raises(TypeError, match=r"Gaussian Proposal Distribution requires a value for sigma"):
        ProposalDistribution("gaussian", sigma=[1., 2.])


def test_it_throws_if_you_specify_multivariate_gaussian_without_values_for_latents() -> None:
    with pytest.raises(TypeError, match=r"Multivariate Gaussian Proposal Distribution requires values for latents"):
        ProposalDistribution("multivariate_gaussian")


def test_it_throws_if_you_specify_multivariate_gaussian_without_a_value_for_sigma(net: BayesNet) -> None:
    with pytest.raises(TypeError, match=r"Multivariate Gaussian Proposal Distribution requires a sigma value for each latent"):
        ProposalDistribution("multivariate_gaussian", latents=list(net.iter_latent_vertices()))


def test_it_throws_if_you_specify_multivariate_gaussian_a_sigma_that_is_not_a_list(net: BayesNet) -> None:
    with pytest.raises(TypeError, match=r"Multivariate Gaussian Proposal Distribution requires a sigma value for each latent"):
        ProposalDistribution("multivariate_gaussian", latents=list(net.iter_latent_vertices()), sigma=np.array([1., 2.]))


def test_it_throws_if_you_specify_multivariate_gaussian_with_not_enough_sigmas_for_each_latent(net: BayesNet) -> None:
    with pytest.raises(TypeError, match=r"Multivariate Gaussian Proposal Distribution requires a sigma value for each latent"):
        ProposalDistribution("multivariate_gaussian", latents=list(net.iter_latent_vertices()), sigma=[1.])


def test_it_throws_if_you_specify_multivariate_gaussian_with_a_list_of_non_tensor_arg_types_as_sigma(net: BayesNet) -> None:
    with pytest.raises(TypeError, match=r"Multivariate Gaussian Proposal Distribution requires a sigma value for each latent"):
        ProposalDistribution("multivariate_gaussian", latents=list(net.iter_latent_vertices()), sigma=[[1.], [2.]])


def test_it_throws_if_you_specify_sigma_but_the_type_isnt_gaussian() -> None:
    with pytest.raises(TypeError, match=r'Parameter sigma is not valid unless type is "gaussian"'):
        ProposalDistribution("prior", sigma=1.)


def test_it_throws_if_it_doesnt_recognise_the_type() -> None:
    with pytest.raises(KeyError, match=r"'foo'"):
        ProposalDistribution("foo")
