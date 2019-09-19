package io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiator;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialsOf;
import lombok.Value;

@Value
public class VariableTransform {

    private final DoubleVertex input;
    private final DoubleVertex output;

    public DoubleTensor transform(DoubleTensor source) {
        input.setAndCascade(source);
        return output.getValue();
    }

    public DoubleTensor dTransform(DoubleTensor source) {
        input.setAndCascade(source);
        PartialsOf partialsOf = Differentiator.reverseModeAutoDiff(output, input);
        return partialsOf.withRespectTo(input);
    }
}
