package io.improbable.keanu.vertices.generic.nonprobabilistic;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import lombok.Value;

@Value
public class CPTCondition {

    private final List<Boolean> conditions;

    public static CPTCondition from(List<Vertex<? extends Tensor<Boolean>>> inputs,
                                    Function<Vertex<? extends Tensor<Boolean>>, Boolean> mapper) {


        List<Boolean> condition = inputs.stream().map(mapper).collect(Collectors.toList());
        return new CPTCondition(condition);
    }
}
