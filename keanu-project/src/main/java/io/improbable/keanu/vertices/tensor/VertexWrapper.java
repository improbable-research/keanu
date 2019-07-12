package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.VertexState;
import lombok.RequiredArgsConstructor;

import java.util.Collection;
import java.util.Optional;
import java.util.Set;

@RequiredArgsConstructor
public class VertexWrapper<TENSOR, VERTEX extends Vertex<TENSOR, VERTEX>> implements Vertex<TENSOR, VERTEX>, NonProbabilistic<TENSOR>, NonSaveableVertex {

    private final NonProbabilisticVertex<TENSOR, VERTEX> vertex;

    public VERTEX setLabel(VertexLabel label) {
        vertex.setLabel(label);
        return (VERTEX) this;
    }

    public VERTEX setLabel(String label) {
        vertex.setLabel(label);
        return (VERTEX) this;
    }

    @Override
    public VertexLabel getLabel() {
        return vertex.getLabel();
    }

    public VERTEX removeLabel() {
        vertex.removeLabel();
        return (VERTEX) this;
    }

    @Override
    public TENSOR lazyEval() {
        return vertex.lazyEval();
    }

    @Override
    public TENSOR eval() {
        return vertex.eval();
    }

    @Override
    public boolean isProbabilistic() {
        return vertex.isProbabilistic();
    }

    @Override
    public boolean isDifferentiable() {
        return vertex.isDifferentiable();
    }

    public void setValue(TENSOR value) {
        vertex.setValue(value);
    }

    @Override
    public TENSOR getValue() {
        return vertex.getValue();
    }

    @Override
    public VertexState<TENSOR> getState() {
        return vertex.getState();
    }

    public void setState(VertexState<TENSOR> newState) {
        vertex.setState(newState);
    }

    @Override
    public boolean hasValue() {
        return vertex.hasValue();
    }

    @Override
    public long[] getShape() {
        return vertex.getShape();
    }

    @Override
    public long[] getStride() {
        return vertex.getStride();
    }

    @Override
    public long getLength() {
        return vertex.getLength();
    }

    @Override
    public int getRank() {
        return vertex.getRank();
    }

    public VERTEX print() {
        vertex.print();
        return (VERTEX) this;
    }

    public VERTEX print(String message, boolean printData) {
        vertex.print(message, printData);
        return (VERTEX) this;
    }

    public void setAndCascade(TENSOR value) {
        vertex.setAndCascade(value);
    }

    public void observe(TENSOR value) {
        vertex.observe(value);
    }

    @Override
    public void observeOwnValue() {
        vertex.observeOwnValue();
    }

    @Override
    public void unobserve() {
        vertex.unobserve();
    }

    @Override
    public boolean isObserved() {
        return vertex.isObserved();
    }

    @Override
    public Optional<TENSOR> getObservedValue() {
        return vertex.getObservedValue();
    }

    @Override
    public VariableReference getReference() {
        return vertex.getReference();
    }

    @Override
    public VertexId getId() {
        return vertex.getId();
    }

    @Override
    public int getIndentation() {
        return vertex.getIndentation();
    }

    @Override
    public Set<Vertex> getChildren() {
        return vertex.getChildren();
    }

    @Override
    public void addChild(Vertex<?, ?> v) {
        vertex.addChild(v);
    }

    @Override
    public void setParents(Collection<? extends Vertex> parents) {
        vertex.setParents(parents);
    }

    @Override
    public void setParents(Vertex<?, ?>... parents) {
        vertex.setParents(parents);
    }

    @Override
    public void addParents(Collection<? extends Vertex> parents) {
        vertex.addParents(parents);
    }

    @Override
    public void addParent(Vertex<?, ?> parent) {
        vertex.addParent(parent);
    }

    @Override
    public Set<Vertex> getParents() {
        return vertex.getParents();
    }

    @Override
    public int getDegree() {
        return vertex.getDegree();
    }

    @Override
    public Set<Vertex> getConnectedGraph() {
        return vertex.getConnectedGraph();
    }

    @Override
    public Class<?> ofType() {
        return vertex.ofType();
    }

    @Override
    public void save(NetworkSaver netSaver) {
        vertex.save(netSaver);
    }

    @Override
    public void saveValue(NetworkSaver netSaver) {
        vertex.saveValue(netSaver);
    }

    @Override
    public void loadValue(NetworkLoader loader) {
        vertex.loadValue(loader);
    }

    @Override
    public TENSOR calculate() {
        return vertex.calculate();
    }

    @Override
    public boolean equals(Object o) {
        return vertex.equals(o);
    }

    @Override
    public int hashCode() {
        return vertex.hashCode();
    }

    @Override
    public String toString() {
        return vertex.toString();
    }

    public NonProbabilisticVertex<TENSOR, VERTEX> getWrappedVertex() {
        return vertex;
    }
}
