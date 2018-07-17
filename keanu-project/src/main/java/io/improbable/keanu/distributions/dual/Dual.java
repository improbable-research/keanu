package io.improbable.keanu.distributions.dual;


import java.util.Objects;

import org.jetbrains.annotations.NotNull;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

/**
 * A Dual is identified only by its name
 * so that you can store it in io.improbable.keanu.distributions.dual.Duals
 */
public class Dual implements Comparable<Dual> {

    private final DualName id;
    private final DoubleTensor value;

    public Dual(DualName name) {
        this(name, null);
    }

    public Dual(DualName name, DoubleTensor value) {
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
        Dual dual = (Dual) o;
        return Objects.equals(id, dual.id);
    }

    @Override
    public int hashCode() {

        return Objects.hash(id);
    }

    @Override
    public int compareTo(@NotNull Dual o) {
        return id.getName().compareTo(o.id.getName());
    }
}
