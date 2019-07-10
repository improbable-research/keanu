package io.improbable.keanu.vertices.intgr;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
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
public class IntegerVertexWrapper implements IntegerVertex, NonProbabilistic<IntegerTensor>, NonSaveableVertex {

    private final NonProbabilisticVertex<IntegerTensor, IntegerVertex> vertex;

    public IntegerVertex setLabel(VertexLabel label) {
        vertex.setLabel(label);
        return this;
    }

    public IntegerVertex setLabel(String label) {
        vertex.setLabel(label);
        return this;
    }

    @Override
    public VertexLabel getLabel() {
        return vertex.getLabel();
    }

    public IntegerVertex removeLabel() {
        vertex.removeLabel();
        return this;
    }

    @Override
    public IntegerTensor lazyEval() {
        return vertex.lazyEval();
    }

    @Override
    public IntegerTensor eval() {
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

    public void setValue(IntegerTensor value) {
        vertex.setValue(value);
    }

    @Override
    public IntegerTensor getValue() {
        return vertex.getValue();
    }

    @Override
    public VertexState<IntegerTensor> getState() {
        return vertex.getState();
    }

    public void setState(VertexState<IntegerTensor> newState) {
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

    public IntegerVertex print() {
        vertex.print();
        return this;
    }

    public IntegerVertex print(String message, boolean printData) {
        vertex.print(message, printData);
        return this;
    }

    public void setAndCascade(IntegerTensor value) {
        vertex.setAndCascade(value);
    }

    public void observe(IntegerTensor value) {
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
    public Optional<IntegerTensor> getObservedValue() {
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
    public IntegerTensor calculate() {
        return vertex.calculate();
    }
}
