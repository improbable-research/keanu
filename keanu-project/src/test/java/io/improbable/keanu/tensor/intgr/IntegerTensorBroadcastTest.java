package io.improbable.keanu.tensor.intgr;

import io.improbable.keanu.tensor.TensorFactories;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;

import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.junit.Assert.assertThat;

@RunWith(Parameterized.class)
public class IntegerTensorBroadcastTest {

    @Parameterized.Parameters(name = "{index}: Test with {1}")
    public static Iterable<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {new Nd4jIntegerTensorFactory(), "ND4J IntegerTensor"},
            {new JVMIntegerTensorFactory(), "JVM IntegerTensor"},
        });
    }

    public IntegerTensorBroadcastTest(IntegerTensorFactory factory, String name) {
        TensorFactories.integerTensorFactory = factory;
    }

    @Test
    public void canBroadcastPow() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, 2, 2);
        IntegerTensor exponent = IntegerTensor.create(2, 3);

        IntegerTensor expected = IntegerTensor.create(new int[]{1, 8, 9, 64}, 2, 2);

        assertThat(matrix.pow(exponent), valuesAndShapesMatch(expected));
        assertThat(matrix.powInPlace(exponent), valuesAndShapesMatch(expected));
    }

}