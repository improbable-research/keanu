package io.improbable.keanu.distributions.dual;

import java.util.Objects;

public class ParameterName {
    private final String name;

    public ParameterName(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ParameterName parameterName = (ParameterName) o;
        return Objects.equals(name, parameterName.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name);
    }

    @Override
    public String toString() {
        return getName();
    }
}
