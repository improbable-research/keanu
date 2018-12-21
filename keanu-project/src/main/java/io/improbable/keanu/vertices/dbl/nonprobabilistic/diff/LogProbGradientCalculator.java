package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import com.google.common.base.Preconditions;
import io.improbable.keanu.network.LambdaSection;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class LogProbGradientCalculator {

    private final Set<? extends Vertex<?>> logProbOfVertices;
    private final Set<? extends Vertex<?>> wrtVertices;

    private final Map<Vertex, Set<DoubleVertex>> parentToLatentLookup;
    private final Map<Vertex, Set<DoubleVertex>> verticesWithNonzeroDiffWrtLatent;

    public LogProbGradientCalculator(List<? extends Vertex> logProbOfVerticesList, List<? extends Vertex<?>> wrtVerticesList) {
        this.logProbOfVertices = new HashSet<>((List<Vertex<?>>) logProbOfVerticesList);
        this.wrtVertices = new HashSet<>(wrtVerticesList);

        parentToLatentLookup = getParentsThatAreConnectedToWrtVertices(logProbOfVertices);
        verticesWithNonzeroDiffWrtLatent = getVerticesWithNonzeroDiffWrt(logProbOfVertices, parentToLatentLookup);
    }

    /**
     * @return the partial derivatives with respect to a given set of latent vertices
     */
    public Map<VertexId, DoubleTensor> getJointLogProbGradientWrtLatents() {
        LogProbGradients totalLogProbGradients = new LogProbGradients();

        for (final Vertex<?> ofVertex : logProbOfVertices) {
            LogProbGradients logProbGradientOfVertex = reverseModeLogProbGradientWrtLatents(ofVertex);
            totalLogProbGradients.add(logProbGradientOfVertex);
        }

        return totalLogProbGradients.getPartials();
    }

    /**
     * The dLogProb(x) method on Vertex returns a partial derivative of the Log Prob with respect to each
     * of its arguments and with respect to its value, x. This method searches these partials for any that
     * are parents of the vertices we are taking the derivative with respect to
     *
     * @param ofVertices          the vertices that the derivative is being calculated "of" with respect to the wrtVertices
     * @param parentToWrtVertices a lookup
     * @return a map for a given vertex to a set of the wrt vertices that it is connected to
     */
    private Map<Vertex, Set<DoubleVertex>> getVerticesWithNonzeroDiffWrt(Set<? extends Vertex<?>> ofVertices, Map<Vertex, Set<DoubleVertex>> parentToWrtVertices) {
        return ofVertices.stream()
            .collect(Collectors.toMap(
                v -> v,
                v -> {
                    Set<DoubleVertex> parents = v.getParents().stream()
                        .map(parent -> (DoubleVertex) parent)
                        .filter(parentToWrtVertices::containsKey)
                        .collect(Collectors.toSet());

                    if (!v.isObserved()) {
                        parents.add((DoubleVertex) v);
                    }

                    return parents;
                }
            ));
    }

    /**
     * This method finds connections between a vertex's parents and any vertices that we are taking the derivative
     * wrt to
     *
     * @param ofVertices the vertices that the derivative is being calculated "of" with respect to the wrtVertices
     * @return a map for a given vertex to a set of vertices that are directly connected to the dLogProb result
     * of the ofVertices and a vertex that we are finding the gradient with respect to.
     */
    private Map<Vertex, Set<DoubleVertex>> getParentsThatAreConnectedToWrtVertices(Set<? extends Vertex> ofVertices) {

        Map<Vertex, Set<DoubleVertex>> probabilisticParentLookup = new HashMap<>();

        for (Vertex<?> probabilisticVertex : ofVertices) {

            Set<? extends Vertex> parents = probabilisticVertex.getParents();

            for (Vertex parent : parents) {

                LambdaSection upstreamLambdaSection = LambdaSection.getUpstreamLambdaSection(parent, false);

                Set<Vertex> latentAndObservedVertices = upstreamLambdaSection.getLatentAndObservedVertices();
                Set<DoubleVertex> latentVertices = latentAndObservedVertices.stream()
                    .filter(this::isLatentDoubleVertexAndInWrtTo)
                    .map(v -> (DoubleVertex) v)
                    .collect(Collectors.toSet());

                if (!latentVertices.isEmpty()) {
                    probabilisticParentLookup.put(parent, latentVertices);
                }
            }
        }

        return probabilisticParentLookup;
    }

    private boolean isLatentDoubleVertexAndInWrtTo(Vertex v) {
        return !v.isObserved() && wrtVertices.contains(v) && v instanceof DoubleVertex;
    }

    /**
     * @param ofVertex the vertex we are taking the derivative of
     * @return partial derivatives of the "ofVertex" wrt to any "this.wrtVertices" that it descends.
     */
    private LogProbGradients reverseModeLogProbGradientWrtLatents(final Vertex ofVertex) {
        Preconditions.checkArgument(
            ofVertex instanceof Probabilistic<?>,
            "Cannot get logProb gradient on non-probabilistic vertex %s", ofVertex
        );

        Set<DoubleVertex> verticesWithNonzeroDiff = verticesWithNonzeroDiffWrtLatent.get(ofVertex);
        final Map<Vertex, DoubleTensor> dlogProbOfVertexWrtVertices = ((Probabilistic<?>) ofVertex).dLogProbAtValue(verticesWithNonzeroDiff);

        LogProbGradients dOfWrtLatentsAccumulated = new LogProbGradients();

        for (Map.Entry<Vertex, DoubleTensor> dlogProbWrtVertex : dlogProbOfVertexWrtVertices.entrySet()) {

            DoubleVertex vertexWithDiff = (DoubleVertex) dlogProbWrtVertex.getKey();
            DoubleTensor dLogProbOfWrtVertexWithDiff = dlogProbWrtVertex.getValue();

            if (vertexWithDiff.equals(ofVertex)) {
                dOfWrtLatentsAccumulated.putWithRespectTo(vertexWithDiff.getId(), dLogProbOfWrtVertexWithDiff);
            } else {

                PartialDerivative partialWrtVertexWithDiff = new PartialDerivative(dLogProbOfWrtVertexWithDiff);

                PartialDerivative correctForScalarReverse = AutoDiffBroadcast.correctForScalarPartialReverse(partialWrtVertexWithDiff, ofVertex.getShape(), vertexWithDiff.getShape());

                PartialsOf dOfWrtLatentsContributionFromParent = Differentiator
                    .reverseModeAutoDiff(
                        vertexWithDiff,
                        correctForScalarReverse,
                        this.parentToLatentLookup.get(vertexWithDiff)
                    );

                dOfWrtLatentsAccumulated = dOfWrtLatentsAccumulated.add(dOfWrtLatentsContributionFromParent.asMap());
            }

        }

        return dOfWrtLatentsAccumulated;
    }

}
