package io.improbable.keanu.distributions.dual;

import java.util.Objects;

import org.jetbrains.annotations.NotNull;

public class ParameterValue<T>  implements Comparable<ParameterValue<T>> {
    private final ParameterName id;
    private final T value;

    public ParameterValue(ParameterName name) {
        this(name, null);
    }

    public ParameterValue(ParameterName name, T value) {
        this.id = name;
        this.value = value;
    }

    public String getName() {
        return id.getName();
    }

    public T getValue() {
        return value;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ParameterValue diff = (ParameterValue) o;
        return Objects.equals(id, diff.id);
    }

    @Override
    public int hashCode() {

        return Objects.hash(id);
    }

    @Override
    public int compareTo(@NotNull ParameterValue o) {
        return id.getName().compareTo(o.id.getName());
    }
}
