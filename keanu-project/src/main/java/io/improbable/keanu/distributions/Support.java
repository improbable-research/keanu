package io.improbable.keanu.distributions;

import lombok.Getter;

public abstract class Support<T> {

    @Getter
    T min;

    @Getter
    T max;

    @Getter
    int[] shape;

    Support(T min, T max, int[] shape) {
        this.min = min;
        this.max = max;
        this.shape = shape;
    }

    public abstract boolean isSubsetOf(Support<T> q);
}
