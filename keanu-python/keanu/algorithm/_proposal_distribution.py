from typing import List, Any, Callable, Dict

from py4j.java_gateway import java_import

from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from keanu.tensor import Tensor
from keanu.vartypes import numpy_types

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.proposal.GaussianProposalDistribution")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.proposal.PriorProposalDistribution")


proposal_distribution_types: Dict[str, Callable] = {
    "gaussian": k.jvm_view().GaussianProposalDistribution,
    "prior": k.jvm_view().PriorProposalDistribution,
}

class ProposalDistribution(JavaObjectWrapper):
    def __init__(self, type_: str, sigma: numpy_types = None, listeners: List[Any] = None) -> None:
        if type_ not in proposal_distribution_types.keys():
            super(ProposalDistribution, self).__init__(proposal_distribution_types["prior"]()) # must construct the object properly before raising the Error
            raise TypeError("Unknown Proposal Distribution type %s" % type_)
        ctor = proposal_distribution_types[type_]
        args = []
        if type_ == "gaussian":
            args.append(Tensor(sigma).unwrap())
        if listeners is not None:
            args.append(listeners)
        super(ProposalDistribution, self).__init__(ctor(*args))