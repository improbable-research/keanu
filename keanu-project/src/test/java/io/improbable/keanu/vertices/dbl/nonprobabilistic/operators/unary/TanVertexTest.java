package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import org.junit.Test;

public class TanVertexTest {

    @Test
    public void canDoTan() {
        TanVertex tan = new TanVertex(Math.PI / 4);
        System.out.println(tan.sample());
    }

    

}
