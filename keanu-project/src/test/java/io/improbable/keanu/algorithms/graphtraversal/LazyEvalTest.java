package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.CeilVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex;
import org.junit.Test;

public class LazyEvalTest {

    @Test
    public void lazyEvalBeingComputedSeveralTimes() {

        ConstantDoubleVertex A = new ConstantDoubleVertex(9.5);
        FloorVertex B = new FloorVertex(A);
        CeilVertex C = new CeilVertex(A);
        AdditionVertex D = new AdditionVertex(B, C);
        FloorVertex E = new FloorVertex(D);
        CeilVertex F = new CeilVertex(D);
        AdditionVertex G = new AdditionVertex(E, F);

        G.lazyEval();

    }

    @Test
    public void random() {
        ConstantDoubleVertex A = new ConstantDoubleVertex(0.5);
        System.out.println(A.getParents());
    }

}
