package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class CPTCondition {
    private final Boolean[] conditions;

    public CPTCondition(Boolean[] condition) {
        this.conditions = condition;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        CPTCondition condition1 = (CPTCondition) o;

        return Arrays.equals(conditions, condition1.conditions);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(conditions);
    }

    public static CPTCondition from(List<Vertex<? extends Tensor<Boolean>>> inputs,
                                    Function<Vertex<? extends Tensor<Boolean>>, Boolean> mapper) {

        Boolean[] condition = new Boolean[inputs.size()];

        for (int i = 0; i < condition.length; i++) {
            condition[i] = mapper.apply(inputs.get(i));
        }

        return new CPTCondition(condition);
    }
}
