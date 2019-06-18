package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.bool.JVMBooleanTensor;
import io.improbable.keanu.tensor.dbl.JVMDoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.tensor.intgr.Nd4jIntegerTensor;
import lombok.AllArgsConstructor;
import lombok.Value;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import static io.improbable.keanu.tensor.TensorShape.getLength;
import static junit.framework.TestCase.assertTrue;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertFalse;

@RunWith(Parameterized.class)
public class BaseTensorTests {

    public interface TensorFactory extends Function<long[], Tensor> {
        Tensor apply(long[] shape);
    }

    @Value
    @AllArgsConstructor
    public static class TensorImpl {
        TensorFactory factory;
        Class clazz;

        public Object[] toObject() {
            return new Object[]{factory, clazz.getSimpleName()};
        }
    }

    @Parameterized.Parameters(name = "{index}: Test with {1}")
    public static Iterable<Object[]> data() {

        List<TensorImpl> tensorImpls = new ArrayList<>();

        tensorImpls.add(new TensorImpl(
            (shape) -> Nd4jDoubleTensor.arange(0, getLength(shape)).reshape(shape), Nd4jDoubleTensor.class
        ));

        tensorImpls.add(new TensorImpl(
            (shape) -> JVMDoubleTensor.arange(0, getLength(shape)).reshape(shape), JVMDoubleTensor.class
        ));

        tensorImpls.add(new TensorImpl(
            (shape) -> Nd4jIntegerTensor.create(1, shape), Nd4jIntegerTensor.class
        ));

        tensorImpls.add(new TensorImpl(
            (shape) -> JVMBooleanTensor.create(true, shape), JVMBooleanTensor.class
        ));

        tensorImpls.add(new TensorImpl(
            (shape) -> GenericTensor.createFilled(1, shape), GenericTensor.class
        ));

        return tensorImpls.stream()
            .map(TensorImpl::toObject)
            .collect(Collectors.toList());
    }

    private final TensorFactory factory;

    public BaseTensorTests(TensorFactory factory, String name) {
        this.factory = factory;
    }

    @Test
    public void rankIsCalculatedCorrectly() {
        assertThat(factory.apply(new long[0]).getRank(), equalTo(0));
        assertThat(factory.apply(new long[]{2}).getRank(), equalTo(1));
        assertThat(factory.apply(new long[]{2, 2}).getRank(), equalTo(2));
        assertThat(factory.apply(new long[]{2, 2, 2}).getRank(), equalTo(3));
        assertThat(factory.apply(new long[]{2, 2, 2, 2}).getRank(), equalTo(4));
    }

    @Test
    public void shapeIsCalculatedCorrectly() {
        assertThat(factory.apply(new long[0]).getShape(), equalTo(new long[0]));
        assertThat(factory.apply(new long[]{2}).getShape(), equalTo(new long[]{2}));
        assertThat(factory.apply(new long[]{2, 2}).getShape(), equalTo(new long[]{2, 2}));
        assertThat(factory.apply(new long[]{2, 2, 2}).getShape(), equalTo(new long[]{2, 2, 2}));
        assertThat(factory.apply(new long[]{2, 2, 2, 2}).getShape(), equalTo(new long[]{2, 2, 2, 2}));
    }

    //    long[] getStride();
    @Test
    public void lengthIsCalculatedCorrectly() {
        assertThat(factory.apply(new long[0]).getLength(), equalTo(1L));
        assertThat(factory.apply(new long[]{2}).getLength(), equalTo(2L));
        assertThat(factory.apply(new long[]{2, 2}).getLength(), equalTo(4L));
        assertThat(factory.apply(new long[]{2, 2, 2}).getLength(), equalTo(8L));
        assertThat(factory.apply(new long[]{2, 2, 2, 2}).getLength(), equalTo(16L));
    }

//     N getValue(long... index)
//
//     void setValue(N value, long... index)

    @Test
    public void getAndSetValueWorks() {

    }

//    N scalar()
//
//    T duplicate();
//
//    T slice(int dimension, long index);
//
//    T take(long... index);

    @Test
    public void canTakeValue() {
        Tensor matrix = factory.apply(new long[]{2, 2});
        assertThat(matrix.getValue(0, 1), equalTo(matrix.take(0, 1).scalar()));
    }
//
//    List<T> split(int dimension, long... splitAtIndices);
//
//    List<T> sliceAlongDimension(int dimension, long indexStart, long indexEnd)
//
//    T diag();
//
//    T transpose()
//
//    N[] asFlatArray();
//
//    T reshape(long... newShape);

    @Test
    public void canReshape() {
        Tensor a = factory.apply(new long[]{2, 2});
        Tensor actual = a.reshape(4);

        assertThat(actual.getShape(), equalTo(new long[]{4}));
        assertThat(actual.getLength(), equalTo(4L));
        assertThat(a.getShape(), equalTo(new long[]{2, 2}));
        assertThat(a.getLength(), equalTo(4L));
    }

//    T permute(int... rearrange);
//
//    T broadcast(long... toShape);
//
//    Tensor.FlattenedView<N> getFlattenedView();
//
//    List<N> asFlatList() {
//        return Arrays.asList(asFlatArray());
//    }
//
//    boolean isLengthOne() {
//        return getLength() == 1;
//    }
//

    @Test
    public void canDetectScalar() {
        assertTrue(factory.apply(new long[0]).isScalar());
        assertFalse(factory.apply(new long[]{2}).isScalar());
    }

    @Test
    public void canDetectVector() {
        assertFalse(factory.apply(new long[0]).isVector());
        assertTrue(factory.apply(new long[]{2}).isVector());
        assertFalse(factory.apply(new long[]{2, 2}).isVector());
    }

    @Test
    public void canDetectMatrix() {
        assertFalse(factory.apply(new long[0]).isMatrix());
        assertFalse(factory.apply(new long[]{2}).isMatrix());
        assertTrue(factory.apply(new long[]{2, 2}).isMatrix());
    }

//    default boolean hasSameShapeAs(Tensor that) {
//        return hasSameShapeAs(that.getShape());
//    }
//
//    default boolean hasSameShapeAs(long[] shape) {
//        return Arrays.equals(this.getShape(), shape);
//    }
//
//    default BooleanTensor elementwiseEquals(Tensor that) {
//        return elementwiseEquals(this, that);
//    }
//
//    BooleanTensor elementwiseEquals(N value);

}
