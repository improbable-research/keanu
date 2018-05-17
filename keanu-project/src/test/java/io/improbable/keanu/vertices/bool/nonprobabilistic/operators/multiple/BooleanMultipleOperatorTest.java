package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import org.junit.Before;
import org.junit.Test;

import java.util.LinkedList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class BooleanMultipleOperatorTest {

    private List<Vertex<Boolean>> allTrue = buildVertexList(15, 15);
    private List<Vertex<Boolean>> allMixed = buildVertexList(18, 10);
    private List<Vertex<Boolean>> allFalse = buildVertexList(15, 0);
    private KeanuRandom random;

    @Before
    public void setup() {
        this.random = new KeanuRandom(1);
    }

    @Test
    public void testMultipleAnd() {
        BoolVertex andTrue = new AndMultipleVertex(allTrue);
        assertEquals(true, andTrue.sample(random));

        BoolVertex andMixed = new AndMultipleVertex(allMixed);
        assertEquals(false, andMixed.sample(random));

        BoolVertex andFalse = new AndMultipleVertex(allFalse);
        assertEquals(false, andFalse.sample(random));
    }

    @Test
    public void testMultipleOr() {
        BoolVertex orTrue = new OrMultipleVertex(allTrue);
        assertEquals(true, orTrue.sample(random));

        BoolVertex orMixed = new OrMultipleVertex(allMixed);
        assertEquals(true, orMixed.sample(random));

        BoolVertex orFalse = new OrMultipleVertex(allFalse);
        assertEquals(false, orFalse.sample(random));
    }

    private List<Vertex<Boolean>> buildVertexList(int numberOfVertices, int numberThatAreTrue) {
        List<Vertex<Boolean>> list = new LinkedList<>();

        for (int i = 0; i < numberThatAreTrue; i++) {
            list.add(new Flip(1.0));
        }

        for (int i = numberThatAreTrue; i < numberOfVertices; i++) {
            list.add(new Flip(0.0));
        }
        return list;
    }

}
