package io.improbable.keanu.vertices;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.network.NetworkSaver;

import java.util.Collection;
import java.util.Optional;
import java.util.Set;

public interface IVertex<T> extends Observable<T>, Variable<T, VertexState<T>> {

    <V extends IVertex<T>> V setLabel(VertexLabel label);

    <V extends IVertex<T>> V setLabel(String label);

    VertexLabel getLabel();

    <V extends IVertex<T>> V removeLabel();

    T lazyEval();

    T eval();

    boolean isProbabilistic();

    boolean isDifferentiable();

    void setValue(T value);

    T getValue();

    VertexState<T> getState();

    void setState(VertexState<T> newState);

    boolean hasValue();

    long[] getShape();

    long[] getStride();

    long getLength();

    int getRank();

    <V extends IVertex<T>> V print();

    <V extends IVertex<T>> V print(final String message, final boolean printData);

    void setAndCascade(T value);

    void observe(T value);

    void observeOwnValue();

    void unobserve();

    boolean isObserved();

    Optional<T> getObservedValue();

    VariableReference getReference();

    VertexId getId();

    int getIndentation();

    Set<IVertex> getChildren();

    void addChild(IVertex<?> v);

    void setParents(Collection<? extends IVertex> parents);

    void setParents(IVertex<?>... parents);

    void addParents(Collection<? extends IVertex> parents);

    void addParent(IVertex<?> parent);

    Set<IVertex> getParents();

    int getDegree();

    Set<IVertex> getConnectedGraph();

    void save(NetworkSaver netSaver);

    void saveValue(NetworkSaver netSaver);

    void loadValue(NetworkLoader loader);
}
