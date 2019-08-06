package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.bool.JVMBooleanTensor;
import io.improbable.keanu.tensor.dbl.JVMDoubleTensor;
import io.improbable.keanu.tensor.dbl.JVMDoubleTensorFactory;
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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;

@RunWith(Parameterized.class)
public class BaseTensorTests {

    public interface TensorFactory extends Function<long[], Tensor<?, ?>> {
        Tensor<?, ?> apply(long[] shape);
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
            (shape) -> JVMDoubleTensorFactory.INSTANCE.arange(0, getLength(shape)).reshape(shape), JVMDoubleTensor.class
        ));

        tensorImpls.add(new TensorImpl(
            (shape) -> Nd4jIntegerTensor.arange(0, (int) getLength(shape)).reshape(shape), Nd4jIntegerTensor.class
        ));

        tensorImpls.add(new TensorImpl(
            (shape) -> Nd4jIntegerTensor.arange(0, (int) getLength(shape)).reshape(shape).mod(2).greaterThan(0), JVMBooleanTensor.class
        ));

        tensorImpls.add(new TensorImpl(
            (shape) -> GenericTensor.create(JVMDoubleTensorFactory.INSTANCE.arange(0, getLength(shape)).asFlatArray(), shape), GenericTensor.class
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

    @Test
    public void lengthIsCalculatedCorrectly() {
        assertThat(factory.apply(new long[0]).getLength(), equalTo(1L));
        assertThat(factory.apply(new long[]{2}).getLength(), equalTo(2L));
        assertThat(factory.apply(new long[]{2, 2}).getLength(), equalTo(4L));
        assertThat(factory.apply(new long[]{2, 2, 2}).getLength(), equalTo(8L));
        assertThat(factory.apply(new long[]{2, 2, 2, 2}).getLength(), equalTo(16L));
    }

    @Test
    public void getAndSetValueWorks() {
        Tensor a = factory.apply(new long[]{2, 2});

        Object v0 = a.getValue(0, 0);

        assertThat(a.asFlatArray()[0], equalTo(v0));

        Object v1 = a.getValue(0, 1);

        assertNotEquals(v0, v1);

        a.setValue(v1, 0, 0);

        assertEquals(a.getValue(0, 0), a.getValue(0, 1));
    }

    @Test
    public void canGetScalar() {
        Tensor a = factory.apply(new long[0]);
        Object scalarValue = a.scalar();
        assertEquals(a.getValue(0), scalarValue);
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsIfNotScalarOnScalar() {
        Tensor a = factory.apply(new long[]{2, 2});
        a.scalar();
    }

    @Test
    public void canDuplicate() {
        Tensor a = factory.apply(new long[]{2, 2});
        Tensor b = a.duplicate();

        b.setValue(b.getValue(0, 1), 0, 0);

        assertEquals(b.getValue(0, 0), b.getValue(0, 1));
        assertNotEquals(a.getValue(0, 0), a.getValue(0, 1));
    }

    @Test
    public void doesDownRankOnSliceRank3To2() {
        Tensor x = factory.apply(new long[]{2, 2, 2});
        TensorTestHelper.doesDownRankOnSliceRank3To2(x);
    }

    @Test
    public void doesDownRankOnSliceRank2To1() {
        Tensor x = factory.apply(new long[]{2, 2});
        TensorTestHelper.doesDownRankOnSliceRank2To1(x);
    }

    @Test
    public void canSliceRank2() {
        Tensor<?, ?> x = factory.apply(new long[]{3, 3});
        Tensor<?, ?> slice = x.slice(1, 0);
        assertThat(slice.getShape(), equalTo(new long[]{3}));
        assertThat(
            slice.asFlatArray(),
            equalTo(
                new Object[]{x.getValue(0, 0), x.getValue(1, 0), x.getValue(2, 0)})
        );
    }

    @Test
    public void doesDownRankOnSliceRank1ToScalar() {
        Tensor x = factory.apply(new long[]{4});
        TensorTestHelper.doesDownRankOnSliceRank1ToScalar(x);
    }

    @Test
    public void canSliceRank1() {
        Tensor<?, ?> x = factory.apply(new long[]{4});
        Tensor<?, ?> slice = x.slice(0, 1);
        assertThat(slice.getShape(), equalTo(new long[0]));
        assertThat(
            slice.asFlatArray(),
            equalTo(
                new Object[]{x.getValue(1)})
        );
    }

    @Test
    public void canTakeValue() {
        Tensor<?, ?> matrix = factory.apply(new long[]{2, 2});
        assertThat(matrix.getValue(0, 1), equalTo(matrix.take(0, 1).scalar()));
    }

    @Test
    public void canSplit() {
        Tensor<?, ?> a = factory.apply(new long[]{2, 2, 2});

        List<? extends Tensor<?, ?>> split = a.split(0, 1, 2);

        assertThat(
            split.get(0).asFlatArray(),
            equalTo(a.slice(0, 0).asFlatArray())
        );

        assertThat(
            split.get(1).asFlatArray(),
            equalTo(a.slice(0, 1).asFlatArray())
        );
    }

    @Test
    public void canSliceAlongDimension() {
        Tensor<?, ?> a = factory.apply(new long[]{2, 2, 2});

        List<? extends Tensor<?, ?>> slices = a.sliceAlongDimension(0, 0, 2);

        assertThat(
            slices.get(0).asFlatArray(),
            equalTo(a.slice(0, 0).asFlatArray())
        );

        assertThat(
            slices.get(1).asFlatArray(),
            equalTo(a.slice(0, 1).asFlatArray())
        );
    }

    @Test
    public void canDiag() {
        Tensor<?, ?> x = factory.apply(new long[]{2});
        Tensor<?, ?> xDiag = x.diag();

        assertThat(xDiag.getShape(), equalTo(new long[]{2, 2}));
        assertThat(xDiag.getValue(0, 0), equalTo(x.getValue(0)));
        assertThat(xDiag.getValue(1, 1), equalTo(x.getValue(1)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsOnDiagOfScalar() {
        Tensor<?, ?> x = factory.apply(new long[0]);
        x.diag();
    }

    @Test
    public void canDiagPart() {
        Tensor<?, ?> x = factory.apply(new long[]{2, 2});

        Tensor<?, ?> diagPart = x.diagPart();
        assertThat(diagPart.getShape(), equalTo(new long[]{2}));
        assertThat(diagPart.getValue(0), equalTo(x.getValue(0, 0)));
        assertThat(diagPart.getValue(1), equalTo(x.getValue(1, 1)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsIfDiagPartOnRank() {
        Tensor<?, ?> x = factory.apply(new long[]{2});
        x.diagPart();
    }

    @Test
    public void permuteDoesCauseReshape() {
        Tensor<?, ?> a = factory.apply(new long[]{2, 1, 3});
        Tensor<?, ?> actual = a.permute(1, 0, 2);
        assertThat(actual.getShape(), equalTo(new long[]{1, 2, 3}));
    }

    @Test
    public void canPermuteUpperDimensions() {
        Tensor<?, ?> a = factory.apply(new long[]{1, 2, 2, 2});

        Tensor<?, ?> permuted = a.permute(0, 1, 3, 2);

        Object expected = new Object[]{
            a.getValue(0), a.getValue(2),
            a.getValue(1), a.getValue(3),
            a.getValue(4), a.getValue(6),
            a.getValue(5), a.getValue(7),
        };

        assertThat(permuted.asFlatArray(), equalTo(expected));
    }

    @Test
    public void canTransposeMatrix() {
        Tensor<?, ?> a = factory.apply(new long[]{2, 2});

        Tensor<?, ?> transposed = a.transpose();

        Object expected = new Object[]{
            a.getValue(0, 0), a.getValue(1, 0),
            a.getValue(0, 1), a.getValue(1, 1),
        };

        assertThat(transposed.asFlatArray(), equalTo(expected));
    }

    @Test(expected = IllegalArgumentException.class)
    public void cannotTransposeVector() {
        Tensor a = factory.apply(new long[]{2});
        a.transpose();
    }

    @Test
    public void canReshape() {
        Tensor<?, ?> a = factory.apply(new long[]{2, 2});
        Tensor<?, ?> actual = a.reshape(4);

        assertThat(actual.getShape(), equalTo(new long[]{4}));
        assertThat(actual.getLength(), equalTo(4L));
        assertThat(a.getShape(), equalTo(new long[]{2, 2}));
        assertThat(a.getLength(), equalTo(4L));
    }

    @Test
    public void canReshapeWithWildCardDim() {
        Tensor a = factory.apply(new long[]{2, 3, 2});

        assertThat(a.reshape(3, -1).getShape(), equalTo(new long[]{3, 4}));
        assertThat(a.reshape(-1, 2).getShape(), equalTo(new long[]{6, 2}));
    }

    @Test
    public void canBroadcastRowVectorToShape() {
        Tensor<?, ?> a = factory.apply(new long[]{1, 2});

        Tensor<?, ?> actual = a.broadcast(2, 2);

        assertThat(actual.getShape(), equalTo(new long[]{2, 2}));
        assertThat(actual.asFlatArray(), equalTo(
            new Object[]{
                a.getValue(0), a.getValue(1),
                a.getValue(0), a.getValue(1)
            })
        );
    }

    @Test
    public void canBroadcastColumnVectorToShape() {
        Tensor<?, ?> a = factory.apply(new long[]{2, 1});

        Tensor<?, ?> actual = a.broadcast(2, 2);

        assertThat(actual.getShape(), equalTo(new long[]{2, 2}));
        assertThat(actual.asFlatArray(), equalTo(
            new Object[]{
                a.getValue(0), a.getValue(0),
                a.getValue(1), a.getValue(1)
            })
        );
    }

    @Test
    public void canBroadcastScalarToShape() {
        Tensor<?, ?> a = factory.apply(new long[0]);

        Tensor<?, ?> actual = a.broadcast(2, 2);

        assertThat(actual.getShape(), equalTo(new long[]{2, 2}));
        assertThat(actual.asFlatArray(), equalTo(
            new Object[]{
                a.getValue(0), a.getValue(0),
                a.getValue(0), a.getValue(0)
            })
        );
    }

    @Test
    public void canGetValueFromFlattenedView() {
        Tensor a = factory.apply(new long[]{2, 2});
        Tensor.FlattenedView flattenedView = a.getFlattenedView();

        assertThat(flattenedView.get(0), equalTo(a.getValue(0, 0)));
        assertThat(flattenedView.get(3), equalTo(a.getValue(1, 1)));

        flattenedView.set(0, flattenedView.get(1));

        assertThat(flattenedView.get(0), equalTo(flattenedView.get(1)));
        assertThat(flattenedView.get(0), equalTo(a.getValue(0, 0)));
    }

    @Test
    public void canGetFlatList() {
        Tensor a = factory.apply(new long[]{2, 2});

        Object[] flatArray = a.asFlatArray();
        Object[] flatList = a.asFlatList().toArray();

        assertThat(flatArray, equalTo(flatList));
    }

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

    @Test
    public void canMoveAxis() {
        /*
         * >>> x = np.zeros((3, 4, 5))
         * >>> np.moveaxis(x, 0, -1).shape
         * (4, 5, 3)
         * >>> np.moveaxis(x, -1, 0).shape
         * (5, 3, 4)
         */

        Tensor x = factory.apply(new long[]{3, 4, 5});
        assertThat(x.moveAxis(0, -1).getShape(), equalTo(new long[]{4, 5, 3}));
        assertThat(x.moveAxis(-1, 0).getShape(), equalTo(new long[]{5, 3, 4}));
    }

    @Test
    public void canSwapAxis() {
        /*
         * >>> x = np.zeros((3, 4, 5))
         * >>> np.swapaxes(x, 0, -1).shape
         * (5, 4, 3)
         */

        Tensor x = factory.apply(new long[]{3, 4, 5});
        assertThat(x.swapAxis(0, -1).getShape(), equalTo(new long[]{5, 4, 3}));
        assertThat(x.swapAxis(1, 2).getShape(), equalTo(new long[]{3, 5, 4}));
    }

    @Test
    public void canSqueeze() {
        Tensor x = factory.apply(new long[]{1, 3, 1});
        assertThat(x.squeeze().getShape(), equalTo(new long[]{3}));
    }

    @Test
    public void canExpandDims() {
        Tensor x = factory.apply(new long[]{3});
        assertThat(x.expandDims(0).getShape(), equalTo(new long[]{1, 3}));
    }

    @Test
    public void canBooleanIndex() {
        Tensor<?, ?> x = factory.apply(new long[]{2, 2});
        Tensor<?, ?> result = x.get(BooleanTensor.create(true, false, false, true).reshape(2, 2));

        assertThat(result.getValue(0), equalTo(x.getValue(0, 0)));
        assertThat(result.getValue(1), equalTo(x.getValue(1, 1)));
        assertThat(result.getLength(), equalTo(2L));
    }

    @Test
    public void canBooleanIndexWithAllFalse() {
        Tensor<?, ?> x = factory.apply(new long[]{2, 2});
        Tensor<?, ?> result = x.get(BooleanTensor.create(false, false, false, false).reshape(2, 2));

        assertThat(result.getLength(), equalTo(0L));
    }

}
