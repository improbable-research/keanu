package io.improbable.keanu.vertices;

import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.algorithms.graphtraversal.DiscoverGraph;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.network.NetworkReader;
import io.improbable.keanu.network.NetworkWriter;
import io.improbable.keanu.tensor.Tensor;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

public abstract class Vertex<T> implements Observable<T>, Samplable<T>, HasShape {

    private final VertexId id = new VertexId();
    private final long[] initialShape;
    private final Observable<T> observation;

    private Set<Vertex> children = Collections.emptySet();
    private Set<Vertex> parents = Collections.emptySet();
    private T value;
    private VertexLabel label = null;

    public Vertex() {
        this(Tensor.SCALAR_SHAPE);
    }

    public Vertex(long[] initialShape) {
        this.initialShape = initialShape;
        this.observation = Observable.observableTypeFor(this.getClass());
    }

    /**
     * Set a label for this vertex.  This allows easy retrieval of this vertex using nothing but a label name.
     *
     * @param label The label to apply to this vertex.  Uniqueness is only enforced on instantiation of a Bayes Net
     * @param <V>   vertex type
     * @return this
     */
    public <V extends Vertex<T>> V setLabel(VertexLabel label) {
        this.label = label;
        return (V) this;
    }

    public <V extends Vertex<T>> V setLabel(String label) {
        return this.setLabel(new VertexLabel(label));
    }

    public VertexLabel getLabel() {
        return this.label;
    }

    public <V extends Vertex<T>> V removeLabel() {
        this.label = null;
        return (V) this;
    }

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
    }

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

    @Override
    public long[] getShape() {
        if (value instanceof Tensor) {
            return ((Tensor) value).getShape();
        } else {
            return initialShape;
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
     * from being observed due to its negative impact on inference algorithms. Non-probabilistic
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

    @Override
    public Optional<T> getObservedValue() {
        return observation.getObservedValue();
    }

    public VertexId getId() {
        return id;
    }

    public String getUniqueStringReference() {
        if (label != null) {
            return label.toString();
        } else {
            return Arrays.stream(id.idValues).boxed()
                .map(Objects::toString)
                .collect(Collectors.joining("_"));
        }
    }

    public int getIndentation() {
        return id.getIndentation();
    }

    public Set<Vertex> getChildren() {
        return children;
    }

    public void addChild(Vertex<?> v) {
        children = ImmutableSet.<Vertex>builder().addAll(children).add(v).build();
    }

    public void setParents(Collection<? extends Vertex> parents) {
        this.parents = Collections.emptySet();
        addParents(parents);
    }

    public void setParents(Vertex<?>... parents) {
        setParents(Arrays.asList(parents));
    }

    public void addParents(Collection<? extends Vertex> parents) {
        this.parents = ImmutableSet.<Vertex>builder().addAll(this.getParents()).addAll(parents).build();
        parents.forEach(p -> p.addChild(this));
    }

    public void addParent(Vertex<?> parent) {
        addParents(ImmutableSet.of(parent));
    }

    public Set<Vertex> getParents() {
        return parents;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Vertex<?> vertex = (Vertex<?>) o;

        return this.id.equals(vertex.id);
    }

    @Override
    public int hashCode() {
        return id.hashCode();
    }

    public Set<Vertex> getConnectedGraph() {
        return DiscoverGraph.getEntireGraph(this);
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(this.getId());
        if (this.getLabel() != null) {
            stringBuilder.append(" (").append(this.getLabel()).append(")");
        }
        stringBuilder.append(": ");
        stringBuilder.append(this.getClass().getSimpleName());
        if (hasValue()) {
            stringBuilder.append("(" + getValue() + ")");
        }
        return stringBuilder.toString();
    }

    public void save(NetworkWriter netWriter) {
        netWriter.save(this);
    }

    public void saveValue(NetworkWriter netWriter) {
        netWriter.saveValue(this);
    }

    public void loadValue(NetworkReader reader) {
       reader.loadValue(this);
    }
}
