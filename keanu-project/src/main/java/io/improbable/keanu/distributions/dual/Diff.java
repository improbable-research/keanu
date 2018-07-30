package io.improbable.keanu.distributions.dual;


import java.util.Objects;

import org.jetbrains.annotations.NotNull;

import io.improbable.keanu.tensor.number.dbl.DoubleTensor;

/**
 * A Diff is identified only by its name
 * so that you can store it in io.improbable.keanu.distributions.dual.Diffs
 */
public class Diff implements Comparable<Diff> {

    private final ParameterName id;
    private final DoubleTensor value;

    public Diff(ParameterName name) {
        this(name, null);
    }

    public Diff(ParameterName name, DoubleTensor value) {
        this.id = name;
        this.value = value;
    }

    public String getName() {
        return id.getName();
    }

    public DoubleTensor getValue() {
        return value;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Diff diff = (Diff) o;
        return Objects.equals(id, diff.id);
    }

    @Override
    public int hashCode() {

        return Objects.hash(id);
    }

    @Override
    public int compareTo(@NotNull Diff o) {
        return id.getName().compareTo(o.id.getName());
    }
}
