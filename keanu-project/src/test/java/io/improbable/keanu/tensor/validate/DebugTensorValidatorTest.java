package io.improbable.keanu.tensor.validate;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.junit.After;
import org.junit.Test;

public class DebugTensorValidatorTest {

    private final DoubleTensor tensor = DoubleTensor.create(1., 2.);
    private final TensorValidator<Double, DoubleTensor> mockValidator = mock(TensorValidator.class);
    private final DebugTensorValidator<Double, DoubleTensor> debugValidator =
            new DebugTensorValidator<>(mockValidator);

    @Test
    public void byDefaultTheValidatorIsOff() {
        debugValidator.check(tensor);
        debugValidator.validate(tensor);
        verifyNoMoreInteractions(mockValidator);
    }

    @After
    public void tearDown() throws Exception {
        debugValidator.disable();
    }

    @Test
    public void youCanEnableChecking() {
        debugValidator.enable();
        debugValidator.check(tensor);
        verify(mockValidator).check(tensor);
        debugValidator.validate(tensor);
        verify(mockValidator).validate(tensor);
        verifyNoMoreInteractions(mockValidator);
    }
}
