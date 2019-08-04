package io.improbable.keanu.backend.keanu.compiled;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.VertexWrapper;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.tensor.number.operators.binary.MultiplicationVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static junit.framework.TestCase.assertEquals;

public class KeanuVertexToTensorOpMapperMutabilityTest {

    private GaussianVertex A;
    private GaussianVertex B;
    private DoubleVertex C;

    @Before
    public void setup() {
        A = new GaussianVertex(0, 1);
        B = new GaussianVertex(0, 1);
        C = A.times(B);
    }

    @Test
    public void doesInPlaceOnLastUseOperationsThatAreMutable() {
        assertInPlace(true, false, true);
    }

    @Test
    public void doesNotInPlaceThatAreImmutable() {
        assertInPlace(false, false, false);
    }

    @Test
    public void doesNotInPlaceNonLastUseOperations() {
        DoubleVertex D = A.plus(B);
        assertInPlace(true, false, false);
    }

    private void assertInPlace(boolean aMutable, boolean bMutable, boolean shouldInPlace) {
        KeanuVertexToTensorOpMapper.OpMapper opMapper = KeanuVertexToTensorOpMapper.getOpMapperFor(MultiplicationVertex.class);

        Map<VariableReference, KeanuCompiledVariable> lookup = new HashMap<>();
        lookup.put(A.getReference(), new KeanuCompiledVariable("A", aMutable));
        lookup.put(B.getReference(), new KeanuCompiledVariable("B", bMutable));

        String result = opMapper.apply(((VertexWrapper<DoubleTensor, DoubleVertex>)C).unwrap(), lookup);

        assertEquals(result.contains("InPlace"), shouldInPlace);
    }

}
