package io.improbable.keanu.vertices.generic.nonprobabilistic;

import java.util.List;
import java.util.function.Function;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import lombok.Value;

@Value
public class CPTCondition {

    private final Boolean[] conditions;

    public static CPTCondition from(List<Vertex<? extends Tensor<Boolean>>> inputs,
                                    Function<Vertex<? extends Tensor<Boolean>>, Boolean> mapper) {

        Boolean[] condition = new Boolean[inputs.size()];

        for (int i = 0; i < condition.length; i++) {
            condition[i] = mapper.apply(inputs.get(i));
        }

        return new CPTCondition(condition);
    }
}
