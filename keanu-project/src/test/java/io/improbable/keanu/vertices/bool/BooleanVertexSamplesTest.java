package io.improbable.keanu.vertices.bool;

import com.google.common.collect.ImmutableList;
import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import static org.hamcrest.MatcherAssert.assertThat;
import org.junit.Test;

import java.util.List;

public class BooleanVertexSamplesTest {
    List<BooleanTensor> values = ImmutableList.of(
        BooleanTensor.create(new boolean[]{true, false}, 1, 2),
        BooleanTensor.create(new boolean[]{true, false}, 1, 2)
    );
    final BooleanVertexSamples samples = new BooleanVertexSamples(values);

    @Test
    public void canGetSamplesAsTensor() {
        BooleanTensor samplesAsTensor = samples.asTensor();
        BooleanTensor expectedTensor = BooleanTensor.create(
            new boolean[] {true, false, true, false},
            2, 1, 2
        );

        assertThat(samplesAsTensor, valuesAndShapesMatch(expectedTensor));
    }

}
