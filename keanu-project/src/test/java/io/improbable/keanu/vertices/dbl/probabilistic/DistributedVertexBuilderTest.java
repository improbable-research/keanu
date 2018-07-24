package io.improbable.keanu.vertices.dbl.probabilistic;

import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.BuilderParameterException;
import io.improbable.keanu.vertices.MissingParameterException;

public class DistributedVertexBuilderTest {

    @Test(expected = BuilderParameterException.class)
    public void itThrowsIfYouPassInTheWrongArgumentTypes() {
        new DistributionVertexBuilder()
            .shaped(1, 2)
            .withInput(ParameterName.MIN, 0.0)
            .withInput(ParameterName.MAX, 1)
            .uniformInt();
    }

    @Test(expected = BuilderParameterException.class)
    public void itThrowsIfYouPassInANullArgument() {
        new DistributionVertexBuilder()
            .shaped(1, 2)
            .withInput(ParameterName.MIN, 0.0)
            .withInput(ParameterName.MAX, (DoubleTensor) null)
            .uniformInt();
    }

    @Test(expected = MissingParameterException.class)
    public void itThrowsIfYouPassInTheWrongParameterNames() {
        new DistributionVertexBuilder()
            .shaped(1, 2)
            .withInput(ParameterName.MIN, 0)
            .withInput(ParameterName.B, 1)
            .uniformInt();
    }

    @Test(expected = MissingParameterException.class)
    public void itThrowsIfYouPassInTooFewParameters() {
        new DistributionVertexBuilder()
            .shaped(1, 2)
            .withInput(ParameterName.MIN, 0)
            .uniformInt();
    }

    @Test(expected = BuilderParameterException.class)
    public void itThrowsIfYouPassInTooManyParameters() {
        new DistributionVertexBuilder()
            .shaped(1, 2)
            .withInput(ParameterName.MIN, 0)
            .withInput(ParameterName.MAX, 1)
            .withInput(ParameterName.B, 1)
            .uniformInt();
    }
}
