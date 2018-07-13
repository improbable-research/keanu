package io.improbable.keanu.vertices;

import java.util.Set;

public interface IVertex<T> {
    public Set<? extends IVertex> getParents();

    boolean hasValue();
    T updateValue();
}
