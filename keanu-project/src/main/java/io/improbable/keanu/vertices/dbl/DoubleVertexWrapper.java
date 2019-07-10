package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.VertexState;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;
import lombok.RequiredArgsConstructor;

import java.util.Collection;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

@RequiredArgsConstructor
public class DoubleVertexWrapper implements DoubleVertex, NonProbabilistic<DoubleTensor>, NonSaveableVertex, Differentiable {

    private final NonProbabilisticVertex<DoubleTensor, DoubleVertex> vertex;

    public DoubleVertex setLabel(VertexLabel label) {
        vertex.setLabel(label);
        return this;
    }

    public DoubleVertex setLabel(String label) {
        vertex.setLabel(label);
        return this;
    }

    @Override
    public VertexLabel getLabel() {
        return vertex.getLabel();
    }

    public DoubleVertex removeLabel() {
        vertex.removeLabel();
        return this;
    }

    @Override
    public DoubleTensor lazyEval() {
        return vertex.lazyEval();
    }

    @Override
    public DoubleTensor eval() {
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

    public void setValue(DoubleTensor value) {
        vertex.setValue(value);
    }

    @Override
    public DoubleTensor getValue() {
        return vertex.getValue();
    }

    @Override
    public VertexState<DoubleTensor> getState() {
        return vertex.getState();
    }

    public void setState(VertexState<DoubleTensor> newState) {
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

    public DoubleVertex print() {
        vertex.print();
        return this;
    }

    public DoubleVertex print(String message, boolean printData) {
        vertex.print(message, printData);
        return this;
    }

    public void setAndCascade(DoubleTensor value) {
        vertex.setAndCascade(value);
    }

    public void observe(DoubleTensor value) {
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
    public Optional<DoubleTensor> getObservedValue() {
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
    public DoubleTensor calculate() {
        return vertex.calculate();
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        return ((Differentiable) vertex).reverseModeAutoDifferentiation(derivativeOfOutputWithRespectToSelf);
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        return ((Differentiable) vertex).forwardModeAutoDifferentiation(derivativeOfParentsWithRespectToInput);
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
}
