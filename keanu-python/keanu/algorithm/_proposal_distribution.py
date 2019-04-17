from typing import List, Any, Callable, Dict, Union

from py4j.java_collections import JavaList, JavaMap
from py4j.java_gateway import java_import, JavaObject

from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from keanu.tensor import Tensor
from keanu.vartypes import tensor_arg_types, runtime_tensor_arg_types
from keanu.vertex import Vertex

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.proposal.GaussianProposalDistribution")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.proposal.PriorProposalDistribution")

proposal_distribution_types: Dict[str, Callable] = {
    "gaussian": k.jvm_view().GaussianProposalDistribution,
    "prior": k.jvm_view().PriorProposalDistribution
}


class ProposalDistribution(JavaObjectWrapper):

    def __init__(self,
                 type_: str,
                 latents: List[Vertex] = None,
                 sigma: Union[tensor_arg_types, List[tensor_arg_types]] = None,
                 default_sigma: tensor_arg_types = None,
                 listeners: List[Any] = []) -> None:
        ctor = proposal_distribution_types[type_]

        if type_ == "gaussian":

            builder = ctor.builder()

            if default_sigma is not None and isinstance(default_sigma, runtime_tensor_arg_types):
                sigma_as_tensor = Tensor(default_sigma).unwrap()
                builder.defaultSigma(sigma_as_tensor.scalar())

            if sigma is not None:

                if isinstance(sigma, list) and 0 < len(latents) == len(sigma) and isinstance(
                        sigma[0], runtime_tensor_arg_types):
                    sigma_as_tensors = [Tensor(s) for s in sigma]
                    latent_and_sigma_pairs = zip(latents, sigma_as_tensors)
                    for pair in latent_and_sigma_pairs:
                        builder.sigmaFor(pair[0].unwrap(), pair[1].unwrap())

                else:
                    raise TypeError("Gaussian Proposal Distribution requires a list of sigmas. One for each latent.")

            if len(listeners) > 0:
                for listener in listeners:
                    builder.proposalListener(listener.unwrap())

            super(ProposalDistribution, self).__init__(builder.build())

        elif type_ == "prior":

            if sigma is not None:
                raise TypeError('Parameter sigma is not valid unless type is "gaussian"')

            if len(listeners) > 0:
                args: List[Union[JavaMap, JavaList, JavaObject]] = [k.to_java_object_list(listeners)]
                super(ProposalDistribution, self).__init__(ctor(*args))
            else:
                super(ProposalDistribution, self).__init__(ctor())
