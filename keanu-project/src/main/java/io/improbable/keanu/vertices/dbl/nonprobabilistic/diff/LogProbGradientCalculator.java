package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import io.improbable.keanu.network.LambdaSection;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class LogProbGradientCalculator {

    private final Set<? extends Vertex> logProbOfVertices;
    private final Set<? extends Vertex> wrtVertices;

    private final Map<Vertex, Set<DoubleVertex>> parentToLatentLookup;
    private final Map<Vertex, Set<DoubleVertex>> verticesWithNonzeroDiffWrtLatent;

    public LogProbGradientCalculator(List<? extends Vertex> logProbOfVerticesList, List<? extends Vertex> wrtVerticesList) {
        this.logProbOfVertices = new HashSet<>(logProbOfVerticesList);
        this.wrtVertices = new HashSet<>(wrtVerticesList);

        parentToLatentLookup = getWrtParents(logProbOfVertices);
        verticesWithNonzeroDiffWrtLatent = getParentsWithNonzeroDiffWrt(logProbOfVertices, parentToLatentLookup);
    }

    private Map<Vertex, Set<DoubleVertex>> getParentsWithNonzeroDiffWrt(Set<? extends Vertex> ofVertices, Map<Vertex, Set<DoubleVertex>> parentToWrtVertices) {
        return ofVertices.stream()
            .collect(Collectors.toMap(
                v -> v,
                v -> {
                    Set<DoubleVertex> parents = ((Set<Vertex>) v.getParents()).stream()
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

    private Map<Vertex, Set<DoubleVertex>> getWrtParents(Set<? extends Vertex> ofVertices) {

        Map<Vertex, Set<DoubleVertex>> probabilisticParentLookup = new HashMap<>();

        for (Vertex probabilisticVertex : ofVertices) {

            Set<? extends Vertex> parents = probabilisticVertex.getParents();

            for (Vertex parent : parents) {

                LambdaSection upstreamLambdaSection = LambdaSection.getUpstreamLambdaSection(parent, false);

                Set<Vertex> latentAndObservedVertices = upstreamLambdaSection.getLatentAndObservedVertices();
                Set<DoubleVertex> latentVertices = latentAndObservedVertices.stream()
                    .filter(v -> !v.isObserved() && v instanceof DoubleVertex)
                    .map(v -> (DoubleVertex) v)
                    .filter(wrtVertices::contains)
                    .collect(Collectors.toSet());

                if (!latentVertices.isEmpty()) {
                    probabilisticParentLookup.put(parent, latentVertices);
                }
            }

        }

        return probabilisticParentLookup;
    }

    /**
     * @return the partial derivatives with respect to any latents upstream
     */
    public Map<VertexId, DoubleTensor> getJointLogProbGradientWrtLatents() {
        PartialDerivatives diffOfLogWrt = new PartialDerivatives(new HashMap<>());

        for (final Vertex<?> ofVertex : logProbOfVertices) {
            diffOfLogWrt = reverseModeLogProbGradientWrtLatents(ofVertex, diffOfLogWrt);
        }

        return diffOfLogWrt.asMap();
    }

    public <T> PartialDerivatives reverseModeLogProbGradientWrtLatents(final Vertex<T> ofVertex,
                                                                       final PartialDerivatives diffOfLogProbWrt) {

        Set<DoubleVertex> verticesWithNonzeroDiff = verticesWithNonzeroDiffWrtLatent.get(ofVertex);
        final Map<Vertex, DoubleTensor> dlogProbOfVertexWrtVertices = ((Probabilistic<T>) ofVertex).dLogProbAtValue(verticesWithNonzeroDiff);

        PartialDerivatives dOfWrtLatents = new PartialDerivatives(new HashMap<>());

        for (Map.Entry<Vertex, DoubleTensor> dlogProbWrtVertex : dlogProbOfVertexWrtVertices.entrySet()) {

            DoubleVertex vertexWithDiff = (DoubleVertex) dlogProbWrtVertex.getKey();
            DoubleTensor dLogProbOfWrtVertexWithDiff = dlogProbWrtVertex.getValue();

            if (vertexWithDiff.equals(ofVertex)) {
                dOfWrtLatents.putWithRespectTo(vertexWithDiff.getId(), dLogProbOfWrtVertexWithDiff);
            } else {

                PartialDerivatives partialWrtVertexWithDiff = new PartialDerivatives(new HashMap<>());
                int[] shapeOfLogProbWrtVertexWithDiff = TensorShape.prependOnes(dLogProbOfWrtVertexWithDiff.getShape(), 2);
                partialWrtVertexWithDiff.putWithRespectTo(vertexWithDiff.getId(), dLogProbOfWrtVertexWithDiff.reshape(shapeOfLogProbWrtVertexWithDiff));

                PartialDerivatives dOfWrtLatentsContributionFromParent = Differentiator
                    .reverseModeAutoDiff(vertexWithDiff, partialWrtVertexWithDiff, this.parentToLatentLookup.get(vertexWithDiff))
                    .sum(true, 0, 1);

                dOfWrtLatents = dOfWrtLatents.add(dOfWrtLatentsContributionFromParent);
            }

        }

        return diffOfLogProbWrt.add(dOfWrtLatents);
    }

}
