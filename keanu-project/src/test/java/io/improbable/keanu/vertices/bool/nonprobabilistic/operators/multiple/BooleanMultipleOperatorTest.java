package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import org.junit.Test;

import java.util.LinkedList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class BooleanMultipleOperatorTest {

    private List<Vertex<BooleanTensor>> allTrue = buildVertexList(15, 15);
    private List<Vertex<BooleanTensor>> allMixed = buildVertexList(18, 10);
    private List<Vertex<BooleanTensor>> allFalse = buildVertexList(15, 0);

    @Test
    public void testMultipleAnd() {
        BooleanVertex andTrue = new AndMultipleVertex(allTrue);
        assertEquals(true, andTrue.eval().scalar());

        BooleanVertex andMixed = new AndMultipleVertex(allMixed);
        assertEquals(false, andMixed.eval().scalar());

        BooleanVertex andFalse = new AndMultipleVertex(allFalse);
        assertEquals(false, andFalse.eval().scalar());
    }

    @Test
    public void testMultipleOr() {
        BooleanVertex orTrue = new OrMultipleVertex(allTrue);
        assertEquals(true, orTrue.eval().scalar());

        BooleanVertex orMixed = new OrMultipleVertex(allMixed);
        assertEquals(true, orMixed.eval().scalar());

        BooleanVertex orFalse = new OrMultipleVertex(allFalse);
        assertEquals(false, orFalse.eval().scalar());
    }

    private List<Vertex<BooleanTensor>> buildVertexList(int numberOfVertices, int numberThatAreTrue) {
        List<Vertex<BooleanTensor>> list = new LinkedList<>();

        for (int i = 0; i < numberThatAreTrue; i++) {
            list.add(new BernoulliVertex(1.0));
        }

        for (int i = numberThatAreTrue; i < numberOfVertices; i++) {
            list.add(new BernoulliVertex(0.0));
        }
        return list;
    }

}
