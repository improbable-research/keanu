package io.improbable.keanu.tensor.intgr;

import static org.hamcrest.MatcherAssert.assertThat;

import static io.improbable.keanu.tensor.TensorMatchers.isScalarWithValue;
import static io.improbable.keanu.tensor.TensorMatchers.tensorEqualTo;

import org.junit.Test;

public class ScalarIntegerTensorTest {
    @Test
    public void canArgFindMaxOfScalar() {
        IntegerTensor tensor = IntegerTensor.scalar(1);

        assertThat(tensor.argMax(), isScalarWithValue(0));
        assertThat(tensor.argMax(0), tensorEqualTo(IntegerTensor.create(0).reshape(1)));
        assertThat(tensor.argMax(1), tensorEqualTo(IntegerTensor.create(0).reshape(1)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void argMaxFailsForAxisTooHigh() {
        IntegerTensor tensor = IntegerTensor.scalar(1);
        tensor.argMax(2);
    }
}
