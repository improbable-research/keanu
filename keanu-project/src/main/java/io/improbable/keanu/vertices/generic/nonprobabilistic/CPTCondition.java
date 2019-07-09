package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.IVertex;
import lombok.Value;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

@Value
public class CPTCondition {

    private final List<?> conditions;

    public static CPTCondition from(List<IVertex<? extends Tensor<?, ?>>> inputs,
                                    Function<IVertex<? extends Tensor<?, ?>>, ?> mapper) {


        List<?> condition = inputs.stream().map(mapper).collect(Collectors.toList());
        return new CPTCondition(condition);
    }
}
