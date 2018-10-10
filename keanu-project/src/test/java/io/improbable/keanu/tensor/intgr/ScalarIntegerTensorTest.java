package io.improbable.keanu.tensor.intgr;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.tensor.TensorMatchers.tensorEqualTo;

import org.junit.Test;

public class ScalarIntegerTensorTest {
    @Test
    public void canArgFindMaxOfScalar() {
        IntegerTensor tensor = IntegerTensor.scalar(1);

        assertEquals(0, tensor.argMax());
        assertThat(tensor.argMax(0), tensorEqualTo(IntegerTensor.scalar(0)));
        assertThat(tensor.argMax(1), tensorEqualTo(IntegerTensor.scalar(0)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void argMaxFailsForAxisTooHigh() {
        IntegerTensor tensor = IntegerTensor.scalar(1);
        tensor.argMax(2);
    }
}
