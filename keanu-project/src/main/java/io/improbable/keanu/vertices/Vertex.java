package io.improbable.keanu.vertices;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

import io.improbable.keanu.algorithms.graphtraversal.DiscoverGraph;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.ValueUpdater;

public abstract class Vertex<T> implements IVertex, Observable<T> {

    public static final AtomicLong ID_GENERATOR = new AtomicLong(0L);

    private long uuid = ID_GENERATOR.getAndIncrement();
    private Set<Vertex> children = new HashSet<>();
    private Set<Vertex> parents = new HashSet<>();
    private T value;
    private final ValueUpdater<T> valueUpdater;
    private final Observable<T> observation;

    public Vertex(ValueUpdater<T> valueUpdater, Observable<T> observation) {
        this.valueUpdater = valueUpdater;
        this.observation = observation;
    }

    /**
     * @param random source of randomness
     * @return a sample from the vertex's distribution. For non-probabilistic vertices,
     * this will always be the same value.
     */
    public abstract T sample(KeanuRandom random);

    public T sample() {
        return sample(KeanuRandom.getDefaultRandom());
    }

    /**
     * This causes a non-probabilistic vertex to recalculate it's value based off it's
     * parent's current values.
     *
     * @return The updated value
     */
    public final T updateValue() {
        if (!valueUpdater.hasValue(this)) {
            setValue(valueUpdater.calculateValue(this));
        }
        return getValue();
    };


    /**
     * This is similar to eval() except it only propagates as far up the graph as required until
     * there are values present to operate on. On a graph that is completely uninitialized,
     * this would be the same as eval()
     *
     * @return the value of the vertex based on the already calculated upstream values
     */
    public final T lazyEval() {
        VertexValuePropagation.lazyEval(this);
        return this.getValue();
    }

    /**
     * This causes a backwards propagating calculation of the vertex value. This
     * propagation only happens for vertices with values dependent on parent values
     * i.e. non-probabilistic vertices. This will also cause probabilistic
     * vertices that have no value to set their value by calling their sample method.
     *
     * @return The value at this vertex after recalculating any parent non-probabilistic
     * vertices.
     */
    public final T eval() {
        VertexValuePropagation.eval(this);
        return this.getValue();
    }

    /**
     * @return True if the vertex is probabilistic, false otherwise.
     * A probabilistic vertex is defined as a vertex whose value is
     * not derived from it's parents. However, the probability of the
     * vertex's value may be dependent on it's parents values.
     */
    public final boolean isProbabilistic() {
        return this instanceof Probabilistic;
    };

    /**
     * Sets the value if the vertex isn't already observed.
     *
     * @param value the observed value
     */
    public void setValue(T value) {
        if (!observation.isObserved()) {
            this.value = value;
        }
    }

    public T getValue() {
        return hasValue() ? value : lazyEval();
    }

    protected T getRawValue() {
        return value;
    }

    public boolean hasValue() {
        if (value instanceof Tensor) {
            return !((Tensor) value).isShapePlaceholder();
        } else {
            return value != null;
        }
    }

    public int[] getShape() {
        if (value instanceof Tensor) {
            return ((Tensor) value).getShape();
        } else {
            return Tensor.SCALAR_SHAPE;
        }
    }

    /**
     * This sets the value in this vertex and tells each child vertex about
     * the new change. This causes a cascading change of values if any of the
     * children vertices are non-probabilistic vertices (e.g. mathematical operations).
     *
     * @param value The new value at this vertex
     */
    public void setAndCascade(T value) {
        setValue(value);
        VertexValuePropagation.cascadeUpdate(this);
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
    @Override
    public void observe(T value) {
        this.value = value;
        observation.observe(value);
    }

    /**
     * Cause this vertex to observe its own value, for example when generating test data
     */
    public void observeOwnValue() {
        observation.observe(getValue());
    }

    @Override
    public void unobserve() {
        observation.unobserve();
    }

    @Override
    public boolean isObserved() {
        return observation.isObserved();
    }

    public boolean matchesObservation() {
        throw new UnsupportedOperationException();
    }

    public long getId() {
        return uuid;
    }

    public Set<Vertex> getChildren() {
        return children;
    }

    public void addChild(Vertex<?> v) {
        children.add(v);
    }

    public void setParents(Collection<? extends Vertex> parents) {
        this.parents = new HashSet<>();
        addParents(parents);
    }

    public void setParents(Vertex<?>... parents) {
        setParents(Arrays.asList(parents));
    }

    public void addParents(Collection<? extends Vertex> parents) {
        parents.forEach(this::addParent);
    }

    public void addParent(Vertex<?> parent) {
        this.parents.add(parent);
        parent.addChild(this);
    }

    @Override
    public Set<? extends IVertex> getParents() {
        return this.parents;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Vertex<?> vertex = (Vertex<?>) o;

        return uuid == vertex.uuid;
    }

    @Override
    public int hashCode() {
        return (int) (uuid ^ (uuid >>> 32));
    }

    public Set<Vertex<?>> getConnectedGraph() {
        return DiscoverGraph.getEntireGraph(this);
    }

}
