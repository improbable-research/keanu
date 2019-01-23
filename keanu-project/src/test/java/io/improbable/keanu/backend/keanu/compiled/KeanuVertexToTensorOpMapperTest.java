package io.improbable.keanu.backend.keanu.compiled;

import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static junit.framework.TestCase.assertEquals;

public class KeanuVertexToTensorOpMapperTest {

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

        String result = opMapper.apply(C, lookup);

        assertEquals(result.contains("InPlace"), shouldInPlace);
    }
}
