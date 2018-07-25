package io.improbable.keanu.vertices;

import java.util.List;

public interface IVertex<T, V extends IVertex<?, V>> {
    public List<? extends V> getParents();

    boolean hasValue();
    T updateValue();

    void setValue(T value);
}
