package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import com.google.common.base.Preconditions;

import io.improbable.keanu.network.LambdaSection;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

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
     * @return the partial derivatives with respect to a give set of latent vertices
     */
    public Map<VertexId, DoubleTensor> getJointLogProbGradientWrtLatents() {
        PartialDerivatives diffOfLogWrt = new PartialDerivatives(new HashMap<>());

        for (final Vertex<?> ofVertex : logProbOfVertices) {
            diffOfLogWrt = diffOfLogWrt.add(reverseModeLogProbGradientWrtLatents(ofVertex));
        }

        return diffOfLogWrt.asMap();
    }

    /**
     * The Vertex::dLogProb method returns a partial derivative of the Log Prob with respect to each parent
     * and with respect to it's own value. This method finds which one of these partials will be non-zero due to being
     * connected to a vertex that we are taking the derivative with respect to.
     *
     * @param ofVertices          the vertices that the derivative is being calculated "of" with respect to the wrtVertices
     * @param parentToWrtVertices a lookup
     * @return a lookup map for a given vertex to a set of the wrt vertices that it descends
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
     * This method finds which one of the vertex's parents are connected to a vertex that we are taking the derivative
     * with respect to.
     *
     * @param ofVertices the vertices that the derivative is being calculated "of" with respect to the wrtVertices
     * @return a lookup map for a given vertex to a set of vertices that are directly connected to the dLogProb result
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
                    .filter(this::isLatentDoubleVertex)
                    .map(v -> (DoubleVertex) v)
                    .collect(Collectors.toSet());

                if (!latentVertices.isEmpty()) {
                    probabilisticParentLookup.put(parent, latentVertices);
                }
            }

        }

        return probabilisticParentLookup;
    }

    private boolean isLatentDoubleVertex(Vertex v) {
        return !v.isObserved() && wrtVertices.contains(v) && v instanceof DoubleVertex;
    }

    /**
     * @param ofVertex starting point for reverse mode autodiff
     * @return partial derivatives of the "ofVertex" wrt to any "this.wrtVertices" that it descends.
     */
    private PartialDerivatives reverseModeLogProbGradientWrtLatents(final Vertex ofVertex) {
        Preconditions.checkArgument(
            ofVertex instanceof Probabilistic<?>,
            "Cannot get logProb gradient on non-probabilistic vertex %s", ofVertex
        );

        Set<DoubleVertex> verticesWithNonzeroDiff = verticesWithNonzeroDiffWrtLatent.get(ofVertex);
        final Map<Vertex, DoubleTensor> dlogProbOfVertexWrtVertices = ((Probabilistic<?>) ofVertex).dLogProbAtValue(verticesWithNonzeroDiff);

        PartialDerivatives dOfWrtLatentsAccumulated = new PartialDerivatives(new HashMap<>());

        for (Map.Entry<Vertex, DoubleTensor> dlogProbWrtVertex : dlogProbOfVertexWrtVertices.entrySet()) {

            DoubleVertex vertexWithDiff = (DoubleVertex) dlogProbWrtVertex.getKey();
            DoubleTensor dLogProbOfWrtVertexWithDiff = dlogProbWrtVertex.getValue();

            if (vertexWithDiff.equals(ofVertex)) {
                dOfWrtLatentsAccumulated.putWithRespectTo(vertexWithDiff.getId(), dLogProbOfWrtVertexWithDiff);
            } else {

                PartialDerivatives partialWrtVertexWithDiff = new PartialDerivatives(new HashMap<>());
                int[] shapeOfLogProbWrtVertexWithDiff = TensorShape.concat(Tensor.SCALAR_SHAPE, dLogProbOfWrtVertexWithDiff.getShape());
                DoubleTensor reshapedToOfWrtPartialFormat = dLogProbOfWrtVertexWithDiff.reshape(shapeOfLogProbWrtVertexWithDiff);
                partialWrtVertexWithDiff.putWithRespectTo(vertexWithDiff.getId(), reshapedToOfWrtPartialFormat);

                PartialDerivatives dOfWrtLatentsContributionFromParent = Differentiator
                    .reverseModeAutoDiff(vertexWithDiff, partialWrtVertexWithDiff, this.parentToLatentLookup.get(vertexWithDiff))
                    .sum(true, 0, 1);

                dOfWrtLatentsAccumulated = dOfWrtLatentsAccumulated.add(dOfWrtLatentsContributionFromParent);
            }

        }

        return dOfWrtLatentsAccumulated;
    }

}
