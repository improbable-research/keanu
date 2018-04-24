package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.LogVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;
import java.util.Random;

import static io.improbable.keanu.kotlin.ExtendPrefixOperatorsKt.exp;
import static org.junit.Assert.assertEquals;

public class DualNumbersTest {

    Random random;
    DoubleVertex vA;
    DoubleVertex vB;

    @Before
    public void setup() {
        random = new Random();
        vA = new GaussianVertex(new ConstantDoubleVertex(1.0), new ConstantDoubleVertex(0.0), random);
        vB = new GaussianVertex(new ConstantDoubleVertex(2.0), new ConstantDoubleVertex(0.0), random);
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
        assertDiffIsCorrect(vA, vB, exp(vA.times(vB)));
    }

    @Test
    public void diffOverPlusMinusMultiplyCombination() {
        DoubleVertex vC = vA.plus(vB);
        DoubleVertex vD = vA.minus(vB);
        DoubleVertex vE = vC.multiply(vD);
        assertDiffIsCorrect(vA, vB, vE);
    }

    @Test
    public void diffOverPlusDivideMultiplyLogCombination() {
        DoubleVertex vC = vA.plus(vB);
        DoubleVertex vD = vA.divideBy(vB);
        DoubleVertex vE = vC.multiply(vD);
        assertDiffIsCorrect(vA, vB, new LogVertex(vE));
    }

    private void assertDiffIsCorrect(DoubleVertex vA, DoubleVertex vB, DoubleVertex vC) {

        double A = 1.0;
        double B = 2.0;

        vA.setValue(A);
        vB.setValue(B);
        vC.lazyEval();

        DualNumber cDual = vC.getDualNumber();

        double C = cDual.getValue();
        Map<String, Double> dc = cDual.getInfinitesimal().getInfinitesimals();

        double da = 0.00000001;

        vA.setValue(vA.getValue() + da);
        vB.setValue(B);
        vC.lazyEval();

        double dcdaApprox = (vC.getValue() - C) / da;

        assertEquals(dcdaApprox, dc.get(vA.getId()), 0.00001);

        double db = da;

        vA.setValue(A);
        vB.setValue(B + db);
        vC.lazyEval();

        double dcdbApprox = (vC.getValue() - C) / db;

        assertEquals(dcdbApprox, dc.get(vB.getId()), 0.00001);
    }
}
