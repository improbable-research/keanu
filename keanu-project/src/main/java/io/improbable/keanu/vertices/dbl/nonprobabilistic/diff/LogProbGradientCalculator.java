package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import io.improbable.keanu.network.LambdaSection;
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
    private final Map<Vertex, Set<DoubleVertex>> parentsWithNonzeroDiffWrtLatent;

    public LogProbGradientCalculator(List<? extends Vertex> logProbOfVerticesList, List<? extends Vertex> wrtVerticesList) {
        this.logProbOfVertices = new HashSet<>(logProbOfVerticesList);
        this.wrtVertices = new HashSet<>(wrtVerticesList);

        parentToLatentLookup = getWrtParents(logProbOfVertices);
        parentsWithNonzeroDiffWrtLatent = getParentsWithNonzeroDiffWrt(logProbOfVertices, parentToLatentLookup);
    }

    private Map<Vertex, Set<DoubleVertex>> getParentsWithNonzeroDiffWrt(Set<? extends Vertex> ofVertices, Map<Vertex, Set<DoubleVertex>> parentToWrtVertices) {
        return ofVertices.stream()
            .collect(Collectors.toMap(
                v -> v,
                v -> ((Set<Vertex>) v.getParents()).stream()
                    .map(parent -> (DoubleVertex) parent)
                    .filter(parentToWrtVertices::containsKey)
                    .collect(Collectors.toSet())
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
                    .filter(v -> !v.isObserved())
                    .map(v -> (DoubleVertex) v)
                    .filter(wrtVertices::contains)
                    .collect(Collectors.toSet());

                probabilisticParentLookup.put(parent, latentVertices);
            }

        }

        return probabilisticParentLookup;
    }

    /**
     * @return the partial derivatives with respect to any latents upstream
     */
    public Map<VertexId, DoubleTensor> getJointLogProbGradientWrtLatents() {
        final Map<VertexId, DoubleTensor> diffOfLogWrt = new HashMap<>();

        for (final Vertex<?> ofVertex : logProbOfVertices) {
            getLogProbGradientWrtLatents(ofVertex, diffOfLogWrt);
        }

        return diffOfLogWrt;
    }

    public <T> Map<VertexId, DoubleTensor> getLogProbGradientWrtLatents(final Vertex<T> ofVertex,
                                                                        final Map<VertexId, DoubleTensor> diffOfLogProbWrt) {


        Set<DoubleVertex> parentsWithNonzeroDiff = parentsWithNonzeroDiffWrtLatent.get(ofVertex);
        final Map<Vertex, DoubleTensor> dlogProbOfVertexWrtParents = ((Probabilistic<T>) ofVertex).dLogProbAtValue(parentsWithNonzeroDiff);

        for (Map.Entry<Vertex, DoubleTensor> dLogProbOfWrtParent : dlogProbOfVertexWrtParents.entrySet()) {

            DoubleVertex parent = (DoubleVertex) dLogProbOfWrtParent.getKey();
            DoubleTensor dLogProbOfwrtParent = dLogProbOfWrtParent.getValue();

            PartialDerivatives dParentWrtLatents = Differentiator.reverseModeAutoDiff(parent, this.parentToLatentLookup.get(parent));

            PartialDerivatives dOfWrtLatents = dParentWrtLatents.multiplyBy(dLogProbOfwrtParent, false);


        }

        return diffOfLogProbWrt;
    }

}
