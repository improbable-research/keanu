package io.improbable.keanu.backend;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.network.LambdaSection;
import io.improbable.keanu.vertices.PlaceholderVertex;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.stream.Collectors;

public interface ComputableGraphBuilder<T extends ComputableGraph> {


    /**
     * @param visiting create this vertex as a constant in the computable graph that is being built.
     */
    void createConstant(Vertex visiting);

    /**
     * @param visiting create this vertex as a variable in the computable graph that is being built.
     */
    void createVariable(Vertex visiting);

    /**
     * @param visiting create this vertex in the compiled graph that is being built.
     */
    void create(Vertex visiting);

    /**
     * @param connections tells the graph builder that any time a vertex is referenced in a connection to reroute it
     *                    to another vertex. This is used to hook up proxy vertices that aren't actually connected to
     *                    parents until they are being added to a graph with this builder.
     */
    void connect(Map<? extends Vertex<?, ?>, ? extends Vertex<?, ?>> connections);

    /**
     * @param output a variable reference to a variable that should be used as an output. This lets the builder know that
     *               the variable should not be mutated and to possibly trim any operations that is not needed for this
     *               output.
     */
    void registerOutput(VariableReference output);

    Collection<VariableReference> getLatentVariables();

    /**
     * Creates an addition operation for the left and right arguments. This is used for adding two graphs that aren't
     * otherwise connected. This happens when summing multiple logProb graphs to form a joint logProb variable
     *
     * @param left  a variable reference to the left arg
     * @param right a variable reference to the right arg
     * @return a new variable reference for the resulting sum
     */
    VariableReference add(VariableReference left, VariableReference right);

    /**
     * @return a computable graph that can be used for a calculation described by the vertices used to build it.
     */
    T build();

    /**
     * @param vertices vertices that represent a graph to create a computable graph from.
     * @param outputs  explicitly specified outputs. This allows the builder to prune operations not needed in order
     *                 to provide the outputs.
     */
    default void convert(Collection<? extends Vertex> vertices, Collection<? extends Vertex> outputs) {

        Set<Vertex> outputsUpstreamLambdaSection = outputs.stream()
            .flatMap(
                output -> LambdaSection
                    .getUpstreamLambdaSection(output, true)
                    .getAllVertices().stream()
            )
            .collect(Collectors.toSet());

        List<? extends Vertex> requiredVertices = vertices.stream()
            .filter(outputsUpstreamLambdaSection::contains)
            .collect(Collectors.toList());

        convert(requiredVertices);
        outputs.stream().map(Vertex::getReference).forEach(this::registerOutput);
    }

    /**
     * @param vertices vertices that represent a graph to create a computable graph from.
     */
    default void convert(Collection<? extends Vertex> vertices) {

        PriorityQueue<Vertex> priorityQueue = new PriorityQueue<>(
            Comparator.comparing(Vertex::getId, Comparator.naturalOrder())
        );

        priorityQueue.addAll(vertices);

        Vertex visiting;
        while ((visiting = priorityQueue.poll()) != null) {

            if (visiting instanceof PlaceholderVertex) {
                continue;
            }

            if (visiting instanceof Probabilistic) {
                if (visiting.isObserved()) {
                    createConstant(visiting);
                } else {
                    createVariable(visiting);
                }
            } else {
                create(visiting);
            }

        }
    }

}
