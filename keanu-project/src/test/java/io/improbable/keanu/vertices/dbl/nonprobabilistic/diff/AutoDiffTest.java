package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.assertEquals;

public class AutoDiffTest {

    DoubleVertex vA;
    DoubleVertex vB;

    @Before
    public void setup() {
        vA = new GaussianVertex(1.0, 0.0);
        vB = new GaussianVertex(2.0, 0.0);
    }

    @Test
    public void diffOverMultiply() {
        assertDiffIsCorrect(vA, vB, vA.multiply(vB));
    }

    @Test
    public void diffOverAddition() {
        assertDiffIsCorrect(vA, vB, vA.plus(vB));
    }

    @Test
    public void diffOverSubtraction() {
        assertDiffIsCorrect(vA, vB, vA.minus(vB));
    }

    @Test
    public void diffOverExponent() {
        assertDiffIsCorrect(vA, vB, vA.multiply(vB).exp());
    }

    @Test
    public void diffOverPlusMinusMultiplyCombination() {
        DoubleVertex vC = vA.plus(vB);
        DoubleVertex vD = vA.minus(vB);
        MultiplicationVertex vE = vC.multiply(vD);
        assertDiffIsCorrect(vA, vB, vE);
    }

    @Test
    public void diffOverPlusDivideMultiplyLogCombination() {
        DoubleVertex vC = vA.plus(vB);
        DoubleVertex vD = vA.divideBy(vB);
        DoubleVertex vE = vC.multiply(vD);
        assertDiffIsCorrect(vA, vB, vE.log());
    }

    private <T extends DoubleVertex & Differentiable> void assertDiffIsCorrect(DoubleVertex vA, DoubleVertex vB, T vC) {

        double A = 1.0;
        double B = 2.0;

        vA.setValue(A);
        vB.setValue(B);
        vC.eval();

        PartialsOf dcdx = Differentiator.reverseModeAutoDiff(vC, vA, vB);

        DoubleTensor C = vC.getValue();
        Map<VertexId, DoubleTensor> dcdxMap = dcdx.asMap();

        double da = 0.00000001;

        vA.setValue(vA.getValue().plus(da));
        vB.setValue(B);
        vC.eval();

        DoubleTensor dcdaApprox = (vC.getValue().minus(C)).div(da);

        assertEquals(dcdaApprox.scalar(), dcdxMap.get(vA.getId()).scalar(), 0.00001);

        double db = da;

        vA.setValue(A);
        vB.setValue(B + db);
        vC.eval();

        DoubleTensor dcdbApprox = (vC.getValue().minus(C)).div(db);

        assertEquals(dcdbApprox.scalar(), dcdxMap.get(vB.getId()).scalar(), 0.00001);
    }
}
