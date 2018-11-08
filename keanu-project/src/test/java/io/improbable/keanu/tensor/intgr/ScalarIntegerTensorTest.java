package io.improbable.keanu.tensor.intgr;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.hasValue;
import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.Assert.assertEquals;

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
        IntegerTensor tensor = IntegerTensor.scalar(1).reshape(1, 1);

        assertEquals(0, tensor.argMax());
        assertThat(tensor.argMax(0), valuesAndShapesMatch(IntegerTensor.scalar(0)));
        assertThat(tensor.argMax(1), valuesAndShapesMatch(IntegerTensor.scalar(0)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void argMaxFailsForAxisTooHigh() {
        IntegerTensor tensor = IntegerTensor.scalar(1);
        tensor.argMax(2);
    }

    @Test
    public void comparesIntegerScalarWithTensor() {
        IntegerTensor value = IntegerTensor.scalar(1);
        IntegerTensor differentValue = IntegerTensor.create(1, 2, 3);
        BooleanTensor result = value.elementwiseEquals(differentValue);
        assertThat(result, hasValue(true, false, false));
    }


    @Test
    public void doesKeepRankOnGTEq() {
        IntegerTensor value = IntegerTensor.create(new int[]{1}, 1, 1, 1);
        assertEquals(3, value.greaterThanOrEqual(2).getRank());
    }

    @Test
    public void doesKeepRankOnGT() {
        IntegerTensor value = IntegerTensor.create(new int[]{1}, 1, 1, 1);
        assertEquals(3, value.greaterThan(2).getRank());
    }

    @Test
    public void doesKeepRankOnLT() {
        IntegerTensor value = IntegerTensor.create(new int[]{1}, 1, 1, 1);
        assertEquals(3, value.lessThan(2).getRank());
    }

    @Test
    public void doesKeepRankOnLTEq() {
        IntegerTensor value = IntegerTensor.create(new int[]{1}, 1, 1, 1);
        assertEquals(3, value.lessThanOrEqual(2).getRank());
    }
}
