package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * This data set was taken from https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
 * It's also the same data set used for scikitlearn load_diabetes at
 * http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
 */
public class DiabetesLinearRegression {

    public static class Data {
        public DoubleTensor bmi;
        public DoubleTensor y;
    }

    @Test
    public void doesLinearRegressionOnBMI() {
        Data data = ReadCsv
            .fromResources("data/datasets/diabetes/diabetes_standardized_training.csv")
            .asVectorizedColumnsDefinedBy(Data.class)
            .load(true);

        // Linear Regression
        DoubleVertex weight = new GaussianVertex(0.0, 2.0);
        DoubleVertex b = new GaussianVertex(0.0, 2.0);
        DoubleVertex x = ConstantVertex.of(data.bmi);
        DoubleVertex yMu = x.multiply(weight);

        DoubleVertex y = new GaussianVertex(yMu.plus(b), 1.0);
        y.observe(data.y);

        BayesianNetwork bayesNet = new BayesianNetwork(y.getConnectedGraph());
        GradientOptimizer optimizer = new GradientOptimizer(bayesNet);
        optimizer.maxLikelihood();

        assertEquals(938.2378, weight.getValue().scalar(), 0.01);
        assertEquals(152.9189, b.getValue().scalar(), 0.01);
    }

}
