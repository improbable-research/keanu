package io.improbable.keanu.tensor.generic;

import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.hasValue;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertThat;

public class GenericTensorTest {
    @Test
    public void canElementwiseEqualsAScalarValue() {
        String value = "foo";
        String otherValue = "bar";
        GenericTensor allTheSame = new GenericTensor(new long[] {2, 3}, value);
        GenericTensor notAllTheSame = allTheSame.duplicate().setValue(otherValue, 1, 1);

        assertThat(allTheSame.elementwiseEquals(value).allTrue(), equalTo(true));
        assertThat(notAllTheSame.elementwiseEquals(value), hasValue(true, true, true, true, false, true));
    }

}