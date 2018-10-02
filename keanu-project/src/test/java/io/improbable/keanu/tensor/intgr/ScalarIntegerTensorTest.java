package io.improbable.keanu.tensor.intgr;

import static org.hamcrest.MatcherAssert.assertThat;

import static io.improbable.keanu.tensor.TensorMatchers.hasValue;

import org.junit.Test;

public class ScalarIntegerTensorTest {

    @Test
    public void canElementwiseEqualsAScalarValue() {
        int value = 42;
        IntegerTensor tensor = IntegerTensor.create(value);

        assertThat(tensor.elementwiseEquals(value), hasValue(true));
        assertThat(tensor.elementwiseEquals(value + 1), hasValue(false));
    }

}