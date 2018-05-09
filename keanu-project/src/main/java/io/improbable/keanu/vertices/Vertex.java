package io.improbable.keanu.vertices;

import io.improbable.keanu.Identifiable;
import io.improbable.keanu.algorithms.graphtraversal.DiscoverGraph;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

public abstract class Vertex<T> implements Identifiable {

    public static final AtomicLong idGenerator = new AtomicLong(0L);

    private String uuid = idGenerator.getAndIncrement() + "";
    private Set<Vertex> children = new HashSet<>();
    private Set<Vertex> parents = new HashSet<>();
    private T value;
    private boolean observed;

    /**
     * This is the natural log of the probability at the supplied value. In the
     * case of continuous vertices, this is actually the log of the density, which
     * will differ from the probability by a constant.
     *
     * @param value The supplied value.
     * @return The natural log of the probability density at the supplied value
     */
    public abstract double logProb(T value);

    public double logProbAtValue() {
        return logProb(getValue());
    }

    /**
     * The partial derivatives of the natural log prob.
     *
     * @param value at a given value
     * @return the partial derivatives of the log density
     */
    public abstract Map<String, DoubleTensor> dLogProb(T value);

    public Map<String, DoubleTensor> dLogProbAtValue() {
        return dLogProb(getValue());
    }

    /**
     * @return a sample from the vertex's distribution. For non-probabilistic vertices,
     * this will always be the same value.
     */
    public abstract T sample();

    /**
     * This causes a non-probabilistic vertex to recalculate it's value based off it's
     * parent's current values.
     *
     * @return The updated value
     */
    public abstract T updateValue();

    /**
     * This causes a backwards propagating calculation of the vertex value. This
     * propagation only happens for vertices with values dependent on parent values
     * i.e. non-probabilistic vertices.
     *
     * @return The value at this vertex after recalculating any parent non-probabilistic
     * vertices.
     */
    public T lazyEval() {
        Deque<Vertex<?>> stack = new ArrayDeque<>();
        stack.push(this);
        Set<Vertex<?>> hasCalculated = new HashSet<>();

        while (!stack.isEmpty()) {

            Vertex<?> head = stack.peek();
            Set<Vertex<?>> parentsThatAreNotYetCalculated = parentsThatAreNotCalculated(hasCalculated, head.getParents());

            if (head.isProbabilistic() || parentsThatAreNotYetCalculated.isEmpty()) {

                Vertex<?> top = stack.pop();
                top.updateValue();
                hasCalculated.add(top);

            } else {

                for (Vertex<?> vertex : parentsThatAreNotYetCalculated) {
                    stack.push(vertex);
                }

            }

        }
        return this.getValue();
    }

    /**
     * @return True if the vertex is probabilistic, false otherwise.
     * A probabilistic vertex is defined as a vertex whose value is
     * not derived from it's parents. However, the probability of the
     * vertex's value may be dependent on it's parents values.
     */
    public abstract boolean isProbabilistic();

    /**
     * Sets the value if the vertex isn't already observed.
     *
     * @param value the observed value
     */
    public void setValue(T value) {
        if (!this.observed) {
            this.value = value;
        }
    }

    public T getValue() {
        return !hasValue() ? lazyEval() : value;
    }

    public boolean hasValue() {
        return value != null;
    }

    /**
     * This sets the value in this vertex and tells each child vertex about
     * the new change. This causes a cascading change of values if any of the
     * children vertices are non-probabilistic vertices (e.g. mathematical operations).
     *
     * @param value The new value at this vertex
     */
    public void setAndCascade(T value) {
        setAndCascade(value, exploreSetting());
    }

    /**
     * @param value    the new value at this vertex
     * @param explored the results of previously exploring the graph, which
     *                 allows the efficient propagation of this new value.
     */
    public void setAndCascade(T value, Map<String, Long> explored) {
        setValue(value);
        VertexValuePropagation.cascadeUpdate(this, explored);
    }

    public Map<String, Long> exploreSetting() {
        return VertexValuePropagation.exploreSetting(this);
    }

    /**
     * This marks the vertex's value as being observed and unchangeable.
     * <p>
     * Non-probabilistic vertices of continuous types (integer, double) are prohibited
     * from being observed due to it's negative impact on inference algorithms. Non-probabilistic
     * booleans are allowed to be observed as well as user defined types.
     *
     * @param value the value to be observed
     */
    public void observe(T value) {
        this.value = value;
        this.observed = true;
    }

    /**
     * Cause this vertex to observe its own value, for example when generating test data
     */
    public void observeOwnValue() {
        this.observed = true;
    }

    public void unobserve() {
        observed = false;
    }

    public boolean isObserved() {
        return observed;
    }

    public String getId() {
        return uuid;
    }

    public Set<Vertex> getChildren() {
        return children;
    }

    public void addChild(Vertex<?> v) {
        children.add(v);
    }

    public void setParents(Collection<? extends Vertex<?>> parents) {
        this.parents = new HashSet<>();
        addParents(parents);
    }

    public void setParents(Vertex<?>... parents) {
        setParents(Arrays.asList(parents));
    }

    public void addParents(Collection<? extends Vertex<?>> parents) {
        parents.forEach(this::addParent);
    }

    public void addParent(Vertex<?> parent) {
        this.parents.add(parent);
        parent.addChild(this);
    }

    public Set<Vertex> getParents() {
        return this.parents;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Vertex<?> vertex = (Vertex<?>) o;

        return uuid.equals(vertex.uuid);
    }

    @Override
    public int hashCode() {
        return uuid.hashCode();
    }

    public Set<Vertex> getConnectedGraph() {
        return DiscoverGraph.getEntireGraph(this);
    }

    private Set<Vertex<?>> parentsThatAreNotCalculated(Set<Vertex<?>> calculated, Set<Vertex> parents) {
        Set<Vertex<?>> notCalculatedParents = new HashSet<>();
        for (Vertex<?> next : parents) {
            if (!calculated.contains(next)) {
                notCalculatedParents.add(next);
            }
        }
        return notCalculatedParents;
    }
}
