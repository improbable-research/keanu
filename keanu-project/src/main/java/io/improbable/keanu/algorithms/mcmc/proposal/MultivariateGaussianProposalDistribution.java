package io.improbable.keanu.algorithms.mcmc.proposal;

import java.util.List;
import java.util.Set;
import java.util.function.Function;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.distributions.continuous.MultivariateGaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Covariance;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class MultivariateGaussianProposalDistribution implements ProposalDistribution {

    private final Covariance covariance;

    public MultivariateGaussianProposalDistribution(Covariance covariance) {
        this.covariance = covariance;
    }

    @Override
    public Proposal getProposal(Set<Vertex> vertices, KeanuRandom random) {
        List<Vertex> sortedVertices = TopologicalSort.sort(vertices);
        DoubleTensor currentValue = createMultivariateValues(sortedVertices, v -> v.getValue());
        VertexId[] vertexIds = vertices.stream().map(Vertex::getId).toArray(VertexId[]::new);
        MultivariateGaussian proposalDistribution = MultivariateGaussian.withParameters(currentValue, covariance.getSubMatrix(vertexIds));
        DoubleTensor result = proposalDistribution.sample(new int[]{1, vertices.size()}, random);

        Proposal proposal = new Proposal();
        for (int i = 0; i < sortedVertices.size(); i++) {
            Vertex<?> vertex = sortedVertices.get(i);
            proposal.setProposal((DoubleVertex) vertex, result.slice(1, i));
        }

        return proposal;
    }

    @Override
    public <T> double logProb(Probabilistic<T> vertex, T ofValue, T givenValue) {
        throw new UnsupportedOperationException("You don't need to call this - just call logProbAtToGivenFrom and logProbAtFromGivenTo");
    }

    @Override
    public double logProbAtToGivenFrom(Proposal proposal) {
        return calculateLogProb(proposal.getVerticesInOrder(), createToValueTensor(proposal), createFromValueTensor(proposal));
    }

    @Override
    public double logProbAtFromGivenTo(Proposal proposal) {
        return calculateLogProb(proposal.getVerticesInOrder(), createFromValueTensor(proposal), createToValueTensor(proposal));
    }

    private double calculateLogProb(List<Vertex> vertices, DoubleTensor atValue, DoubleTensor givenValue) {
        VertexId[] vertexIds = vertices.stream().map(Vertex::getId).toArray(VertexId[]::new);
        return MultivariateGaussian.withParameters(givenValue, covariance.getSubMatrix(vertexIds)).logProb(atValue).sum();
    }

    private DoubleTensor createFromValueTensor(Proposal proposal) {
        return createMultivariateValues(proposal.getVerticesInOrder(), proposal::getProposalFrom);
    }

    private DoubleTensor createToValueTensor(Proposal proposal) {
        return createMultivariateValues(proposal.getVerticesInOrder(), proposal::getProposalTo);
    }

    private DoubleTensor createMultivariateValues(List<Vertex> vertices, Function<Vertex<DoubleTensor>, DoubleTensor> valueGetter) {
        ImmutableList.Builder<DoubleTensor> builder = ImmutableList.builder();
        for (Vertex vertex : vertices) {
            builder.add(valueGetter.apply(vertex));
        }
        ImmutableList<DoubleTensor> values = builder.build();
        return DoubleTensor.concat(0, values.toArray(new DoubleTensor[0]));
    }
}
