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
                    .filter(v -> !v.isObserved())
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
            diffOfLogWrt = getLogProbGradientWrtLatents(ofVertex, diffOfLogWrt);
        }

        return diffOfLogWrt.asMap();
    }

    public <T> PartialDerivatives getLogProbGradientWrtLatents(final Vertex<T> ofVertex,
                                                               final PartialDerivatives diffOfLogProbWrt) {


        Set<DoubleVertex> parentsWithNonzeroDiff = parentsWithNonzeroDiffWrtLatent.get(ofVertex);
        final Map<Vertex, DoubleTensor> dlogProbOfVertexWrtParents = ((Probabilistic<T>) ofVertex).dLogProbAtValue(parentsWithNonzeroDiff);

        PartialDerivatives dOfWrtLatents = new PartialDerivatives(new HashMap<>());

        for (Map.Entry<Vertex, DoubleTensor> diffWrtParent : dlogProbOfVertexWrtParents.entrySet()) {

            DoubleVertex parent = (DoubleVertex) diffWrtParent.getKey();
            DoubleTensor dLogProbOfWrtParent = diffWrtParent.getValue().reshape(TensorShape.prependOnes(diffWrtParent.getValue().getShape(), 2));

            if (parentToLatentLookup.containsKey(parent) && !parentToLatentLookup.get(parent).isEmpty()) {

                PartialDerivatives dLogProbdParent = new PartialDerivatives(new HashMap<>());
                dLogProbdParent.putWithRespectTo(parent.getId(), dLogProbOfWrtParent);

                PartialDerivatives dOfWrtLatentsContributionFromParent = Differentiator
                    .reverseModeAutoDiff(parent, dLogProbdParent, this.parentToLatentLookup.get(parent))
                    .sum(true, TensorShape.dimensionRange(0, 2));

                dOfWrtLatents = dOfWrtLatents.add(dOfWrtLatentsContributionFromParent);
            } else {
                dOfWrtLatents.putWithRespectTo(parent.getId(), dLogProbOfWrtParent.sum(TensorShape.dimensionRange(0, 2)));
            }

        }

        return diffOfLogProbWrt.add(dOfWrtLatents);
    }

}
