package io.improbable.keanu.tensor.intgr;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.tensor.TensorMatchers.hasValue;
import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;

import org.junit.Test;

public class ScalarIntegerTensorTest {

    @Test
    public void canElementwiseEqualsAScalarValue() {
        int value = 42;
        IntegerTensor tensor = IntegerTensor.create(value);

        assertThat(tensor.elementwiseEquals(value), hasValue(true));
        assertThat(tensor.elementwiseEquals(value + 1), hasValue(false));
    }

    @Test
    public void canArgFindMaxOfScalar() {
        IntegerTensor tensor = IntegerTensor.scalar(1);

        assertEquals(0, tensor.argMax());
        assertThat(tensor.argMax(0), valuesAndShapesMatch(IntegerTensor.scalar(0)));
        assertThat(tensor.argMax(1), valuesAndShapesMatch(IntegerTensor.scalar(0)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void argMaxFailsForAxisTooHigh() {
        IntegerTensor tensor = IntegerTensor.scalar(1);
        tensor.argMax(2);
    }
}
