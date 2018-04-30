package io.improbable.keanu.vertices;

import io.improbable.keanu.Identifiable;
import io.improbable.keanu.algorithms.graphtraversal.DiscoverGraph;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

public abstract class Vertex<T> implements Identifiable {

    public static AtomicLong idGenerator = new AtomicLong(0L);

    private String uuid = idGenerator.getAndIncrement() + "";
    private Set<Vertex<?>> children = new HashSet<>();
    private Set<Vertex<?>> parents = new HashSet<>();
    private T value;
    private boolean observed;

    /**
     * This is the value of the probability density at the supplied value.
     *
     * @param value The supplied value.
     * @return The probability.
     */
    public abstract double density(T value);

    /**
     * Just a helper method for a common function
     */
    public double densityAtValue() {
        return density(getValue());
    }

    /**
     * This is the value of the natural log of the probability density at the supplied value.
     *
     * @param value The supplied value.
     * @return The probability.
     */
    public double logDensity(T value) {
        return Math.log(density(value));
    }

    /**
     * Just a helper method for a common function
     */
    public double logDensityAtValue() {
        return logDensity(getValue());
    }

    /**
     * This returns the derivative of the density function with respect to
     * any dependent vertices.
     *
     * @return a Map containing { dependent vertex Id -&gt; density slope w.r.t. dependent vertex}
     */
    public abstract Map<String, Double> dDensityAtValue();

    /**
     * This is the same as dDensityAtValue except for the log of the density. For numerical
     * stability a vertex may chose to override this method but if not overridden, the
     * chain rule is used to calculate the derivative of the log of the density.
     * <p>
     * dlog(P)/dx = (dP/dx)*(1/P(x))
     */
    public Map<String, Double> dlnDensityAtValue() {

        final double density = densityAtValue();
        Map<String, Double> dDensityAtValue = dDensityAtValue();
        Map<String, Double> dLnDensity = new HashMap<>();
        for (String vertexId : dDensityAtValue.keySet()) {
            dLnDensity.put(vertexId, dDensityAtValue.get(vertexId) / density);
        }

        return dLnDensity;
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
     * A probabilistic vertex is defined as a vertex whose value is not
     * derived from it's parents. However, the probability of the vertex's
     * value may be dependent on it's parents values.
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

    public Set<Vertex<?>> getChildren() {
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

    public Set<Vertex<?>> getParents() {
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

    public Set<Vertex<?>> getConnectedGraph() {
        return DiscoverGraph.getEntireGraph(this);
    }

    private Set<Vertex<?>> parentsThatAreNotCalculated(Set<Vertex<?>> calculated, Set<Vertex<?>> parents) {
        Set<Vertex<?>> notCalculatedParents = new HashSet<>();
        for (Vertex<?> next : parents) {
            if (!calculated.contains(next)) {
                notCalculatedParents.add(next);
            }
        }
        return notCalculatedParents;
    }
}
