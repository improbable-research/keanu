package io.improbable.keanu.vertices;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import static io.improbable.keanu.tensor.TensorMatchers.hasShape;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class SamplableWithShapeTest {

    public static final long[] SHAPE = {2, 3, 4};

    public SamplableWithManyScalars<BooleanTensor> mockSamplable(long[] shape) {
        SamplableWithManyScalars<BooleanTensor> samplable = mock(SamplableWithManyScalars.class);
        when(samplable.getShape()).thenReturn(shape);
        when(samplable.sample()).thenCallRealMethod();
        when(samplable.sample(any(KeanuRandom.class))).thenCallRealMethod();
        when(samplable.sampleManyScalars(any(long[].class))).thenCallRealMethod();
        when(samplable.sampleManyScalars(any(long[].class), any(KeanuRandom.class))).thenCallRealMethod();
        when(samplable.sampleWithShape(any(long[].class), any(KeanuRandom.class)))
            .thenAnswer(invocation -> {
                long[] newShape = invocation.getArgument(0);
                return BooleanTensor.placeHolder(newShape);
            });
        return samplable;
    }

    @Test
    public void sampleUsesShapeOfTensor() {
        SamplableWithManyScalars<BooleanTensor> samplable = mockSamplable(SHAPE);
        BooleanTensor result = samplable.sample();
        verify(samplable).sampleWithShape(SHAPE, KeanuRandom.getDefaultRandom());
        assertThat(result, hasShape(samplable.getShape()));
    }

    @Test
    public void sampleManyScalarsUsesInputShape() {
        SamplableWithManyScalars<BooleanTensor> samplable = mockSamplable(Tensor.SCALAR_SHAPE);
        BooleanTensor result = samplable.sampleManyScalars(SHAPE);
        verify(samplable).sampleWithShape(SHAPE, KeanuRandom.getDefaultRandom());
        assertThat(result, hasShape(SHAPE));
    }

    @Test(expected = IllegalArgumentException.class)
    public void cannotSampleManyScalarsOfNonScalar() {
        SamplableWithManyScalars<BooleanTensor> samplable = mockSamplable(SHAPE);
        samplable.sampleManyScalars(SHAPE);
    }
}
