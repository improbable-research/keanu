package io.improbable.keanu.util;

import io.improbable.keanu.vertices.tensor.If;
import io.improbable.keanu.vertices.tensor.bool.BooleanVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic.BinomialVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.hamcrest.core.AnyOf;
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

        assertThat(description, AnyOf.anyOf(
            is("This Vertex = BinomialVertex(p=Const(2.0), n=Three)"),
            is("This Vertex = BinomialVertex(n=Three, p=Const(2.0))"))
        );
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
    public void testDoubleIfVertexDescriptionCreatedCorrectly() {
        BooleanVertex predicate = new ConstantBooleanVertex(false);

        DoubleVertex three = new ConstantDoubleVertex(3.0).setLabel("Three");
        DoubleVertex four = new ConstantDoubleVertex(4.0).setLabel("Four");

        DoubleVertex result = If.isTrue(predicate).then(three).orElse(four);
        String description = descriptionCreator.createDescription(result);

        assertThat(description, is("This Vertex = Const(false) ? Three : Four"));
    }

    @Test
    public void testIntegerIfVertexDescriptionCreatedCorrectly() {
        BooleanVertex predicate = new ConstantBooleanVertex(false);

        IntegerVertex three = new ConstantIntegerVertex(3).setLabel("Three");
        IntegerVertex four = new ConstantIntegerVertex(4).setLabel("Four");

        IntegerVertex result = If.isTrue(predicate).then(three).orElse(four);
        String description = descriptionCreator.createDescription(result);

        assertThat(description, is("This Vertex = Const(false) ? Three : Four"));
    }

    @Test
    public void testBooleanIfVertexDescriptionCreatedCorrectly() {
        BooleanVertex predicate = new ConstantBooleanVertex(false);

        BooleanVertex trueVertex = new ConstantBooleanVertex(true).setLabel("True Label");
        BooleanVertex falseVertex = new ConstantBooleanVertex(false).setLabel("False Label");

        BooleanVertex result = If.isTrue(predicate).then(trueVertex).orElse(falseVertex);
        String description = descriptionCreator.createDescription(result);

        assertThat(description, is("This Vertex = Const(false) ? True Label : False Label"));
    }

    @Test
    public void usesBracketsForNestedIfs() {
        BooleanVertex predicate1 = new ConstantBooleanVertex(false);

        BooleanVertex trueVertex = new ConstantBooleanVertex(true).setLabel("True Label");
        BooleanVertex falseVertex = new ConstantBooleanVertex(false).setLabel("False Label");

        BooleanVertex path1 = If.isTrue(predicate1).then(trueVertex).orElse(falseVertex);

        BooleanVertex result = If.isTrue(predicate1).then(path1).orElse(falseVertex);
        String description = descriptionCreator.createDescription(result);

        assertThat(description, is("This Vertex = Const(false) ? (Const(false) ? True Label : False Label) : False Label"));
    }

    @Test
    public void testBooleanUnaryOpsDescriptionsCreatedCorrectly() {
        DoubleVertex two = new ConstantDoubleVertex(2.0);
        DoubleVertex three = new ConstantDoubleVertex(3.0).setLabel("three");

        BooleanVertex falze = new ConstantBooleanVertex(false);

        BooleanVertex predicate1 = two.greaterThanOrEqual(three);
        BooleanVertex predicate2 = two.greaterThan(three);
        BooleanVertex predicate3 = two.lessThanOrEqual(three);
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

    @Test
    public void usesVertexLabelWhenProvided() {
        IntegerVertex two = new ConstantIntegerVertex(2).setLabel("Two");
        IntegerVertex three = new ConstantIntegerVertex(3).setLabel("Three");

        IntegerVertex result = two.multiply(three).setLabel("Result");
        String description = descriptionCreator.createDescription(result);

        assertThat(description, is("Result = Two * Three"));
    }

    @Test
    public void createsNullDescription() {
        String description = descriptionCreator.createDescription(null);

        assertThat(description, is("This Vertex = Null"));
    }
}
