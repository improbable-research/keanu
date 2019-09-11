package io.improbable.keanu.vertices;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.hasShape;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class SamplableWithShapeTest {

    public static final long[] SHAPE = {2, 3, 4};

    public Samplable<BooleanTensor> mockSamplable(long[] shape) {
        Samplable<BooleanTensor> samplable = mock(Samplable.class);
        when(samplable.getShape()).thenReturn(shape);
        when(samplable.sample()).thenCallRealMethod();
        when(samplable.sample(any(KeanuRandom.class))).thenCallRealMethod();
        when(samplable.batchSample(any(long[].class))).thenCallRealMethod();
        when(samplable.batchSample(any(long[].class), any(KeanuRandom.class))).thenCallRealMethod();
        when(samplable.sample(any(long[].class), any(KeanuRandom.class)))
            .thenAnswer(invocation -> {
                long[] newShape = invocation.getArgument(0);
                return BooleanTensor.create(true, newShape);
            });
        return samplable;
    }

    @Test
    public void sampleUsesShapeOfTensor() {
        Samplable<BooleanTensor> samplable = mockSamplable(SHAPE);
        BooleanTensor result = samplable.sample();
        verify(samplable).sample(SHAPE, KeanuRandom.getDefaultRandom());
        assertThat(result, hasShape(samplable.getShape()));
    }

    @Test
    public void sampleManyScalarsUsesInputShape() {
        Samplable<BooleanTensor> samplable = mockSamplable(Tensor.SCALAR_SHAPE);
        BooleanTensor result = samplable.batchSample(SHAPE);
        verify(samplable).sample(SHAPE, KeanuRandom.getDefaultRandom());
        assertThat(result, hasShape(SHAPE));
    }

    @Test(expected = IllegalArgumentException.class)
    public void cannotSampleManyScalarsOfNonScalar() {
        Samplable<BooleanTensor> samplable = mockSamplable(SHAPE);
        samplable.batchSample(SHAPE);
    }
}
