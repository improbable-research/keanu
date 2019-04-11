package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import lombok.Value;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

@Value
public class CPTCondition<T> {

    private final List<T> conditions;

    public static <T> CPTCondition<T> from(List<Vertex<? extends Tensor<T>>> inputs,
                                    Function<Vertex<? extends Tensor<T>>, T> mapper) {


        List<T> condition = inputs.stream().map(mapper).collect(Collectors.toList());
        return new CPTCondition<>(condition);
    }
}
