from typing import List, Any, Callable, Dict, Iterable

from py4j.java_gateway import java_import

from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from keanu.tensor import Tensor
from keanu.vartypes import numpy_types
from keanu.vertex import Vertex

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.proposal.GaussianProposalDistribution")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.proposal.PriorProposalDistribution")

proposal_distribution_types: Dict[str, Callable] = {
    "gaussian": k.jvm_view().GaussianProposalDistribution,
    "prior": k.jvm_view().PriorProposalDistribution,
}


class ProposalDistribution(JavaObjectWrapper):

    def __init__(self,
                 type_: str,
                 latents: Iterable[Vertex] = None,
                 sigma: numpy_types = None,
                 listeners: List[Any] = []) -> None:
        ctor = proposal_distribution_types[type_]
        args = []

        if type_ == "gaussian":
            if sigma is None:
                raise TypeError("Gaussian Proposal Distribution requires a value for sigma")
            args.append(Tensor(sigma).unwrap())
        elif sigma is not None:
            raise TypeError('Parameter sigma is not valid unless type is "gaussian"')

        if type_ == "prior":
            if latents is None:
                raise TypeError("Prior Proposal Distribution requires latent variables")

        if len(listeners) > 0:
            args.append(k.to_java_object_list(listeners))
        super(ProposalDistribution, self).__init__(ctor(*args))
