package io.improbable.keanu.distributions.dual;

import java.util.Objects;

public class DualName {
    private final String name;

    public DualName(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        DualName dualName = (DualName) o;
        return Objects.equals(name, dualName.name);
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
