package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.intgr.Nd4jIntegerTensorFactory;
import io.improbable.keanu.tensor.lng.JVMLongTensorFactory;
import io.improbable.keanu.tensor.lng.LongTensor;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.function.Function;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

@RunWith(Parameterized.class)
public class FixedPointTensorTest<N extends Number> {

    @Parameterized.Parameters(name = "{index}: Test with {2}")
    public static Iterable<Object[]> data() {
        Function<Long, Long> toLong = in -> in;
        Function<Long, Integer> toInt = Long::intValue;

        return Arrays.asList(new Object[][]{
            {new JVMLongTensorFactory(), toLong, "JVM LongTensor"},
            {new Nd4jIntegerTensorFactory(), toInt, "Nd4j IntegerTensor"}
        });
    }

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    private FixedPointTensorFactory<N, ?> factory;
    private Function<Long, N> typed;

    public FixedPointTensorTest(FixedPointTensorFactory<N, ?> factory, Function<Long, N> typed, String name) {
        this.factory = factory;
        this.typed = typed;
    }

    private N typed(long in) {
        return typed.apply(in);
    }

    @Test
    public void canMod() {
        FixedPointTensor<N, ?> value = factory.create(4, 5);

        assertThat(value.mod(typed(3)).toLong(), equalTo(LongTensor.create(1, 2)));
        assertThat(value.mod(typed(2)).toLong(), equalTo(LongTensor.create(0, 1)));
        assertThat(value.mod(typed(4)).toLong(), equalTo(LongTensor.create(0, 1)));
    }
}
