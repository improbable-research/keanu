package io.improbable.keanu.e2e.regression;


import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * This data set was taken from https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
 * It's also the same data set used for scikitlearn load_diabetes at
 * http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
 */
public class DiabetesLinearRegression {

    public static void main(String[] args) {
        List<Data> data = ReadCsv
            .fromResources("data/datasets/diabetes/diabetes.csv")
            .as(Data.class)
            .asList(true);

        doModel(data);
    }

    public static void doModel(List<Data> data) {

        int size = data.size();
        double[] ages = new double[size];
        double[] sexs = new double[size];
        double[] bmis = new double[size];
        double[] bps = new double[size];
        double[] s1s = new double[size];
        double[] s2s = new double[size];
        double[] s3s = new double[size];
        double[] s4s = new double[size];
        double[] s5s = new double[size];
        double[] s6s = new double[size];
        double[] ys = new double[size];

        for (int i = 0; i < data.size(); i++) {
            Data d = data.get(i);
            ages[i] = d.age;
            sexs[i] = d.sex;
            bmis[i] = d.bmi;
            bps[i] = d.bp;
            s1s[i] = d.s1;
            s2s[i] = d.s2;
            s3s[i] = d.s3;
            s4s[i] = d.s4;
            s5s[i] = d.s5;
            s6s[i] = d.s6;
            ys[i] = d.y;
        }

        DoubleTensor ageTensor = DoubleTensor.create(ages);
        DoubleTensor sexTensor = DoubleTensor.create(sexs);
        DoubleTensor bmiTensor = DoubleTensor.create(bmis);
        DoubleTensor bpTensor = DoubleTensor.create(bps);
        DoubleTensor s1Tensor = DoubleTensor.create(s1s);
        DoubleTensor s2Tensor = DoubleTensor.create(s2s);
        DoubleTensor s3Tensor = DoubleTensor.create(s3s);
        DoubleTensor s4Tensor = DoubleTensor.create(s4s);
        DoubleTensor s5Tensor = DoubleTensor.create(s5s);
        DoubleTensor s6Tensor = DoubleTensor.create(s6s);
        DoubleTensor yData = DoubleTensor.create(ys);

        DoubleTensor[] xData = new DoubleTensor[]{
            ageTensor,
            sexTensor,
            bmiTensor,
            bpTensor,
            s1Tensor,
            s2Tensor,
            s3Tensor,
            s4Tensor,
            s5Tensor,
            s6Tensor
        };

        // Linear Regression
        DoubleVertex[] weights = new DoubleVertex[xData.length];
        DoubleVertex[] x = new DoubleVertex[xData.length];
        DoubleVertex yMu = ConstantVertex.of(0.0);
        for (int i = 0; i < weights.length; i++) {
            weights[i] = new GaussianVertex(0.0, 2.0);
            weights[i].setValue(0);
            x[i] = ConstantVertex.of(xData[i]);
            yMu = yMu.plus(x[i].multiply(weights[i]));
        }

        DoubleVertex b = new UniformVertex(-50, 50);
        b.setValue(0);
        yMu = yMu.plus(b);

        DoubleVertex y = new GaussianVertex(yMu, 1.0);
        y.observe(yData);

        BayesianNetwork bayesNet = new BayesianNetwork(y.getConnectedGraph());
        GradientOptimizer optimizer = new GradientOptimizer(bayesNet);
        optimizer.maxLikelihood(10000);

        for (int i = 0; i < xData.length; i++) {
            System.out.println(weights[i].getValue().scalar());
        }
    }

    public static class Data {
        public double age;
        public double sex;
        public double bmi;
        public double bp;
        public double s1;
        public double s2;
        public double s3;
        public double s4;
        public double s5;
        public double s6;
        public double y;
    }
}
