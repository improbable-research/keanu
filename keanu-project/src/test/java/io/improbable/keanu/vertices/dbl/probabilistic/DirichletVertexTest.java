package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class DirichletVertexTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void sampleFromUnivariateReturnsFlatDistribution() {
        DirichletVertex dirichlet = new DirichletVertex(2.0);

        Assert.assertEquals(1.0, dirichlet.sample(random).scalar(), 1e-6);
        Assert.assertEquals(1.0, dirichlet.logPdf(0.5), 1e-6);
    }

    @Test
    public void flatDirichletIfAllConcentrationAreOnes() {
        DirichletVertex dirichlet = new DirichletVertex(new ConstantDoubleVertex(new double[]{1, 1}));

        Assert.assertEquals(0.0, dirichlet.logPdf(DoubleTensor.create(new double[]{1.3, 1.6})), 1e-6);
        Assert.assertEquals(0.0, dirichlet.logPdf(DoubleTensor.create(new double[]{0.3, 0.6})), 1e-6);
        Assert.assertEquals(0.0, dirichlet.logPdf(DoubleTensor.create(new double[]{30, 50})), 1e-6);
    }

    @Test
    public void splitStrings() {
        DirichletVertex dirichlet = new DirichletVertex(new ConstantDoubleVertex(new double[]{10, 5, 3}));
        int numSamples = 50000;
        DoubleTensor samples = Nd4jDoubleTensor.zeros(new int[]{numSamples, 3});

        for (int i = 0; i < numSamples; i++) {
            DoubleTensor sample = dirichlet.sample(random);
            samples.setValue(sample.getValue(0, 0), i, 0);
            samples.setValue(sample.getValue(0, 1), i, 1);
            samples.setValue(sample.getValue(0, 2), i, 2);
        }

        DoubleTensor stringOne = samples.slice(1, 0);
        DoubleTensor stringTwo = samples.slice(1, 1);
        DoubleTensor stringThree = samples.slice(1, 2);

        double stringOneLength = stringOne.average();
        double stringTwoLength = stringTwo.average();
        double stringThreeLength = stringThree.average();

        Assert.assertEquals(1.0, stringOneLength + stringTwoLength + stringThreeLength, 1e-3);
        Assert.assertEquals(10. / 18., stringOneLength, 1e-3);
        Assert.assertEquals(5. / 18., stringTwoLength, 1e-3);
        Assert.assertEquals(3. / 18., stringThreeLength, 1e-3);

    }

    @Test
    public void dog() {
        DirichletVertex d = new DirichletVertex(new ConstantDoubleVertex(new double[]{2, 2, 2}));
        System.out.println(d.sample(random));
        System.out.println(d.logPdf(new double[]{0.5, 0.5}));
    }

}
