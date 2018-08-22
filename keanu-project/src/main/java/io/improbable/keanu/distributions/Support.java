package io.improbable.keanu.distributions;

import io.improbable.keanu.tensor.Tensor;
import lombok.Getter;

import java.util.Arrays;

public abstract class Support<T extends Tensor> {

    @Getter
    T min;

    @Getter
    T max;

    @Getter
    int[] shape;

    Support(T min, T max, int[] shape) {
        if (!Arrays.equals(min.getShape(), max.getShape())) {
            throw new IllegalArgumentException("min and max must have same shape");
        }

        this.min = min;
        this.max = max;
        this.shape = shape;
    }

    public abstract boolean isSubsetOf(Support<T> q);
}
