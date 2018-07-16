package io.improbable.keanu.vertices;

import java.util.Set;

public interface IVertex<T, V extends IVertex<?, V>> {
    public Set<? extends V> getParents();

    boolean hasValue();
    T updateValue();

    void setValue(T value);
}
