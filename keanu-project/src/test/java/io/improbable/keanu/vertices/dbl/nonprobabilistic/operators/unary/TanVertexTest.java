package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

public class TanVertexTest {

    @Test
    public void tanTest() {
        GaussianVertex unknownAngle = new GaussianVertex(0.0, 1.0);

        TanVertex tan = new TanVertex(unknownAngle);
        SinVertex sin = new SinVertex(unknownAngle);

        DoubleVertex x = tan.div(sin);
        GaussianVertex g = new GaussianVertex(x, 0.1);
        g.observe(1.15470053838);

        BayesNet bayesNet = new BayesNet(tan.getConnectedGraph());
        GradientOptimizer gradientOptimizer = new GradientOptimizer(bayesNet);
        gradientOptimizer.maxLikelihood(1000);

        System.out.println(unknownAngle.getValue());
    }

    @Test
    public void tanIdentity() {
        GaussianVertex x = new GaussianVertex(0, 1);
        GaussianVertex y = new GaussianVertex(0, 1);
        GaussianVertex z = new GaussianVertex(0, 1);

        TanVertex tanX = new TanVertex(x);
        TanVertex tanY = new TanVertex(y);
        TanVertex tanZ = new TanVertex(z);

        DoubleVertex sum = tanX.plus(tanY).plus(tanZ);
        DoubleVertex product = tanX.multiply(tanY).multiply(tanZ);

        GaussianVertex sumGaussian = new GaussianVertex(sum, 0.1);
        GaussianVertex sumProduct = new GaussianVertex(product, 0.1);

        double value = 1.73205080757;

        sumGaussian.observe(value);
        sumProduct.observe(value);

        BayesNet bayesNet = new BayesNet(x.getConnectedGraph());
        GradientOptimizer gradientOptimizer = new GradientOptimizer(bayesNet);

        gradientOptimizer.maxLikelihood(1000);

        System.out.println("Pi = " + Math.PI);
        System.out.println("Sum of angles = " + (x.getValue() + y.getValue() + z.getValue()));
    }

}
