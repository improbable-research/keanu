package io.improbable.keanu.util;

import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.BinomialVertex;
import org.junit.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.core.Is.is;

public class DescriptionGeneratorTest {

    DescriptionCreator descriptionCreator = new DescriptionCreator();

    @Test
    public void testBinomialVertexDescriptionCreatedCorrectly() {
        DoubleVertex two = new ConstantDoubleVertex(2.0);
        IntegerVertex three = new ConstantIntegerVertex(3).setLabel("Three");
        BinomialVertex binomialVertex = new BinomialVertex(two, three);

        String description = descriptionCreator.createDescription(binomialVertex);

        assertThat(description, is("This Vertex = BinomialVertex(p=Const(2.0), n=Three)"));
    }

    @Test
    public void testSimpleDescriptionsCreatedCorrectly() {
        DoubleVertex two = new ConstantDoubleVertex(2.0);
        DoubleVertex three = new ConstantDoubleVertex(3.0).setLabel("Three");
        DoubleVertex four = new ConstantDoubleVertex(4.0).setLabel("Four");

        DoubleVertex result = two.multiply(three).minus(four);
        String description = descriptionCreator.createDescription(result);

        assertThat(description, is("This Vertex = (Const(2.0) * Three) - Four"));
    }

    @Test
    public void testIntegerAdditionAndMultiplicationDescriptionsCreatedCorrectly() {
        IntegerVertex two = new ConstantIntegerVertex(2);
        IntegerVertex three = new ConstantIntegerVertex(3).setLabel("Three");
        IntegerVertex four = new ConstantIntegerVertex(4).setLabel("Four");

        IntegerVertex result = two.multiply(three).plus(four);
        String description = descriptionCreator.createDescription(result);

        assertThat(description, is("This Vertex = (Const(2) * Three) + Four"));
    }

    @Test
    public void testIfVertexDescriptionCreatedCorrectly() {
        BooleanVertex predicate = new ConstantBooleanVertex(false);

        DoubleVertex three = new ConstantDoubleVertex(3.0).setLabel("Three");
        DoubleVertex four = new ConstantDoubleVertex(4.0).setLabel("Four");

        DoubleVertex result = If.isTrue(predicate).then(three).orElse(four);
        String description = descriptionCreator.createDescription(result);

        assertThat(description, is("This Vertex = Const(false) ? Three : Four"));
    }

    @Test
    public void testBooleanUnaryOpsDescriptionsCreatedCorrectly() {
        DoubleVertex two = new ConstantDoubleVertex(2.0);
        DoubleVertex three = new ConstantDoubleVertex(3.0).setLabel("three");

        BooleanVertex falze = new ConstantBooleanVertex(false);

        BooleanVertex predicate1 = two.greaterThanOrEqualTo(three);
        BooleanVertex predicate2 = two.greaterThan(three);
        BooleanVertex predicate3 = two.lessThanOrEqualTo(three);
        BooleanVertex predicate4 = two.lessThan(three);

        BooleanVertex predicate5 = falze.or(falze);
        BooleanVertex predicate6 = falze.and(falze);

        assertThat(descriptionCreator.createDescription(predicate1), is("This Vertex = Const(2.0) >= three"));
        assertThat(descriptionCreator.createDescription(predicate2), is("This Vertex = Const(2.0) > three"));
        assertThat(descriptionCreator.createDescription(predicate3), is("This Vertex = Const(2.0) <= three"));
        assertThat(descriptionCreator.createDescription(predicate4), is("This Vertex = Const(2.0) < three"));
        assertThat(descriptionCreator.createDescription(predicate5), is("This Vertex = Const(false) || Const(false)"));
        assertThat(descriptionCreator.createDescription(predicate6), is("This Vertex = Const(false) && Const(false)"));
    }

    @Test
    public void testSingleConstVertexDescription() {
        IntegerVertex vertex = new ConstantIntegerVertex(1);
        IntegerVertex vertexWithShape = new ConstantIntegerVertex(new int[]{1, 1});

        String description = descriptionCreator.createDescription(vertex);
        String descriptionWithShape = descriptionCreator.createDescription(vertexWithShape);

        assertThat(description, is("This Vertex = Const(1)"));
        assertThat(descriptionWithShape, is("This Vertex = ConstantIntegerVertex with shape: [2]"));
    }
}
