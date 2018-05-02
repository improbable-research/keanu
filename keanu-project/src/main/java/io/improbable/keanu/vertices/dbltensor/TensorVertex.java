package io.improbable.keanu.vertices.dbltensor;

import java.util.*;

public abstract class TensorVertex<TENSOR extends Tensor> {

    private String uuid = UUID.randomUUID().toString();
    private Set<TensorVertex> children = new HashSet<>();
    private Set<TensorVertex> parents = new HashSet<>();
    private TENSOR value;
    private boolean observed;

    /**
     * This is the value of the probability density at the current value.
     *
     * @return The probability.
     */
    public abstract double density();

    /**
     * This is the value of the natural log of the probability density at the current value.
     *
     * @return The probability.
     */
    public double logDensity() {
        return Math.log(density());
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

        final double density = density();
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
    public abstract TENSOR sample();

    /**
     * This causes a non-probabilistic vertex to recalculate it's value based off it's
     * current parent values.
     *
     * @return The updated value
     */
    public abstract TENSOR updateValue();

    /**
     * This causes a backwards propagating calculation of the vertex value. This
     * propagation only happens for vertices with values dependent on parent values
     * i.e. non-probabilistic vertices.
     *
     * @return The value at this vertex after recalculating any parent non-probabilistic
     * vertices.
     */
    public abstract TENSOR lazyEval();

    /**
     * A probabilistic vertex is defined as a vertex that is probabilistic.
     */
    public abstract boolean isProbabilistic();

    /**
     * Sets the value if the vertex isn't already observed.
     *
     * @param value the observed value
     */
    public void setValue(TENSOR value) {
        if (!this.observed) {
            this.value = value;
        }
    }

    public TENSOR getValue() {
        return value == null ? lazyEval() : value;
    }

    /**
     * This sets the value in this vertex and tells each child vertex about
     * the new change. This causes a cascading change of values if any of the
     * children vertices are lambda vertices.
     *
     * @param value The new value at this vertex
     */
    public void setAndCascade(TENSOR value) {
        setValue(value);
        updateChildren();
    }

    /**
     * This causes this vertex's value to propagate to child vertices.
     */
    public void updateChildren() {
        for (TensorVertex child : this.children) {
            child.updateValue();
            child.updateChildren();
        }
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
    public void observe(TENSOR value) {
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

    public Set<TensorVertex> getChildren() {
        return children;
    }

    public void addChild(TensorVertex v) {
        children.add(v);
    }

    public void setParents(Collection<? extends TensorVertex> parents) {
        this.parents = new HashSet<>();
        addParents(parents);
    }

    public void setParents(TensorVertex... parents) {
        setParents(Arrays.asList(parents));
    }

    public void addParents(Collection<? extends TensorVertex> parents) {
        parents.forEach(this::addParent);
    }

    public void addParent(TensorVertex parent) {
        this.parents.add(parent);
        parent.addChild(this);
    }

    public Set<TensorVertex> getParents() {
        return this.parents;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        TensorVertex vertex = (TensorVertex) o;

        return uuid.equals(vertex.uuid);
    }

    @Override
    public int hashCode() {
        return uuid.hashCode();
    }
}

