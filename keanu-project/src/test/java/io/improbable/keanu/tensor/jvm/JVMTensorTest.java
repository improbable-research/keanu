package io.improbable.keanu.tensor.jvm;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.JVMDoubleTensorFactory;
import org.junit.Test;

import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertThat;

public class JVMTensorTest {

    @Test
    public void canToStringRank3() {
        DoubleTensor matrix = JVMDoubleTensorFactory.INSTANCE.arange(0, 8).reshape(2, 2, 2);
        String actual = matrix.toString();

        assertThat(actual, equalTo(
            "{\n" +
                "shape = [2, 2, 2]\n" +
                "data = \n" +
                "[[[0.0, 1.0],\n" +
                "  [2.0, 3.0]],\n" +
                "\n" +
                " [[4.0, 5.0],\n" +
                "  [6.0, 7.0]]]\n" +
                "}"));
    }

    @Test
    public void canToStringMatrix() {
        DoubleTensor matrix = JVMDoubleTensorFactory.INSTANCE.arange(0, 9).reshape(3, 3);
        String actual = matrix.toString();

        assertThat(actual, equalTo(
            "{\n" +
                "shape = [3, 3]\n" +
                "data = \n" +
                "[[0.0, 1.0, 2.0],\n" +
                " [3.0, 4.0, 5.0],\n" +
                " [6.0, 7.0, 8.0]]\n}"));
    }

    @Test
    public void canToStringVector() {
        DoubleTensor vector = JVMDoubleTensorFactory.INSTANCE.arange(0, 3);
        String actual = vector.toString();

        assertThat(actual, equalTo(
            "{\n" +
                "shape = [3]\n" +
                "data = \n" +
                "[0.0, 1.0, 2.0]\n" +
                "}"));
    }

    @Test
    public void canToStringScalar() {
        DoubleTensor scalar = JVMDoubleTensorFactory.INSTANCE.scalar(9);
        String actual = scalar.toString();

        assertThat(actual, equalTo(
            "{\n" +
                "shape = []\n" +
                "data = \n9.0\n}"));
    }
}
