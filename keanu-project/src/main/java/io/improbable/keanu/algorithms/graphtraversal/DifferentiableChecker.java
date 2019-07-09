package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import lombok.experimental.UtilityClass;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * Utility class for checking whether the given vertices are all differentiable w.r.t latents.
 * When given latent variables, this ensures that the dLogProb can be calculated.
 * <p>
 * This check is performed by traversing up each vertex's parents and ensuring that the path to next RV is
 * differentiable or constant valued.
 * If there is a non differentiable vertex on this path, then if it is constant valued (0 gradient) it has no effect
 * and therefore will return true.
 * <p>
 * -- Examples --
 * RV = Random Variable
 * (G) = A vertex we want to check whether differentiable
 * ND = Non-differentiable vertex
 * D = Differentiable vertex
 * C = Constant valued vertex
 * <p>
 * - Differentiable -
 * <p>
 *      RV  RV
 *       \  /
 *        D    RV
 *         \  /
 *         RV(G)
 * <p>
 * This graph is differentiable as traversing up each of the vertex's parent to the next RV is a differentiable path.
 * <p>
 *      C    C
 *       \  /
 *        ND    RV
 *         \  /
 *         RV(G)
 * <p>
 * This graph is differentiable as the path that is non differentiable is constant valued.
 * <p>
 * - Not Differentiable -
 * <p>
 *      RV  RV     - Both RV not observed
 *       \  /
 *        ND    RV
 *         \  /
 *         RV(G)
 * <p>
 * This is not differentiable as there is a non differentiable path which does not have a constant value.
 * <p>
 * For more examples see DifferentiableCheckerTest.java.
 */
@UtilityClass
public class DifferentiableChecker {

    /**
     * @param vertices the vertices to check are differentiable w.r.t latents.
     * @return true if all given vertices are differentiable, false otherwise.
     */
    public static boolean isDifferentiableWrtLatents(Collection<IVertex> vertices) {
        // All probabilistic need to be double or observed to ensure that the dLogProb can be calculated, for example
        // the dLogProb of BernoulliVertex can only be calculated when it is observed.
        if (!allProbabilisticAreDoubleOrObserved(vertices)) {
            return false;
        }
        Set<IVertex> allParents = allParentsOf(vertices);
        Set<IVertex> constantValueVerticesCache = new HashSet<>();
        return diffableOrConstantUptoNextRV(allParents, constantValueVerticesCache);
    }

    private static boolean allProbabilisticAreDoubleOrObserved(Collection<IVertex> vertices) {
        return vertices.stream().filter(IVertex::isProbabilistic)
            .allMatch(DifferentiableChecker::isDoubleOrObserved);
    }

    private static boolean isDoubleOrObserved(IVertex v) {
        return (v instanceof DoubleVertex || v.isObserved());
    }

    private static Set<IVertex> allParentsOf(Collection<IVertex> vertices) {
        Set<IVertex> allParents = new HashSet<>();
        for (IVertex vertex : vertices) {
            allParents.addAll(vertex.getParents());
        }
        return allParents;
    }

    private static boolean diffableOrConstantUptoNextRV(Collection<IVertex> vertices, Set<IVertex> constantValueVerticesCache) {
        return BreadthFirstSearch.bfsWithFailureCondition(vertices,
            vertex -> isNonDiffableAndNotConstant(vertex, constantValueVerticesCache),
            DifferentiableChecker::getParentsIfVertexIsNotProbabilistic,
            BreadthFirstSearch::doNothing);
    }

    private static Collection<IVertex> getParentsIfVertexIsNotProbabilistic(IVertex visiting) {
        return visiting.isProbabilistic() ? Collections.emptySet() : visiting.getParents();
    }

    private static boolean isNonDiffableAndNotConstant(IVertex vertex, Set<IVertex> constantValueVerticesCache) {
        return !vertex.isDifferentiable() &&
            !isVertexValueConstant(vertex, constantValueVerticesCache);
    }

    private static boolean isVertexValueConstant(IVertex vertex, Set<IVertex> constantValueVerticesCache) {
        if (isValueKnownToBeConstant(vertex, constantValueVerticesCache)) {
            return true;
        }

        return BreadthFirstSearch.bfsWithFailureCondition(Collections.singletonList(vertex),
            DifferentiableChecker::isUnobservedProbabilistic,
            visiting -> getParentsIfValueNotKnownToBeConstant(visiting, constantValueVerticesCache),
            constantValueVerticesCache::addAll);
    }

    private static boolean isUnobservedProbabilistic(IVertex vertex) {
        return vertex.isProbabilistic() && !vertex.isObserved();
    }

    private static Collection<IVertex> getParentsIfValueNotKnownToBeConstant(IVertex visiting,
                                                                             Set<IVertex> constantValueVerticesCache) {

        return isValueKnownToBeConstant(visiting, constantValueVerticesCache) ? Collections.emptySet() : visiting.getParents();
    }

    // We know whether these are constant. For cases such as a MultiplicationVertex we would need to
    // explore its parents to ensure its constant.
    private static boolean isValueKnownToBeConstant(IVertex vertex, Set<IVertex> constantValueVerticesCache) {
        return vertex instanceof ConstantVertex || vertex.isObserved() || constantValueVerticesCache.contains(vertex);
    }
}
