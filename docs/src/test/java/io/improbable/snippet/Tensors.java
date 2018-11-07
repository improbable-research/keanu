package io.improbable.snippet;


import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertexFactory;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Ignore;

import static org.junit.Assert.assertArrayEquals;

public class Tensors {

    @Ignore
    public void doesSimpleTensorInferenceExample() {

//%%SNIPPET_START%% TensorExample
DoubleVertex muA = ConstantVertexFactory.of(new double[]{0.5, 1.5});
DoubleVertex A = new GaussianVertex(new long[]{1, 2}, muA, 1);
DoubleVertex B = ConstantVertexFactory.of(new double[]{3, 4});

DoubleVertex C = A.times(B);
DoubleVertex CObservation = new GaussianVertex(C, 1);
CObservation.observe(new double[]{6, 12});

//Use algorithm to find MAP or posterior samples for A and/or B
Optimizer optimizer = Optimizer.of(new BayesianNetwork(A.getConnectedGraph()));
optimizer.maxLikelihood();
//%%SNIPPET_END%% TensorExample

assertArrayEquals(new double[]{2.0, 3.0}, A.getValue().asFlatDoubleArray(), 1e-2);
    }

    private static void tensorExamples1() {
//%%SNIPPET_START%% TensorSharedValue
DoubleTensor dTensor = DoubleTensor.create(5, new long[]{1, 4});     //[5, 5, 5, 5]
IntegerTensor iTensor = IntegerTensor.create(1, new long[]{1, 4});    //[1, 1, 1, 1]
BooleanTensor bTensor = BooleanTensor.create(true, new long[]{1, 4}); //[true, true, true, true]
//%%SNIPPET_END%% TensorSharedValue
    }

    private static void tensorExamples2() {
//%%SNIPPET_START%% Tensor2by2
DoubleTensor dTensor = DoubleTensor.create(new double[]{0.5, 1.5, 2.5, 3.5}, new long[]{2, 2});
IntegerTensor iTensor = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
BooleanTensor bTensor = BooleanTensor.create(new boolean[]{true, true, false, false}, new long[]{2, 2});
//%%SNIPPET_END%% Tensor2by2
    }

    private static void tensorExample3() {
//%%SNIPPET_START%% TensorReshape
DoubleTensor tensor = DoubleTensor.create(new double[]{0.5, 1.5, 2.5, 3.5}, new long[]{2, 2});
tensor.getShape();       //[2, 2]
tensor.reshape(1, 4);
tensor.getShape();       //[1, 4]
//%%SNIPPET_END%% TensorReshape
    }

    private static void tensorOps() {
//%%SNIPPET_START%% TensorOps
DoubleTensor tensor = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});
tensor.plus(1.0);           // [2, 3, 4, 5]
tensor.times(2.0);          // [4, 6, 8, 10]
tensor.pow(2);              // [16, 36, 64, 100]
tensor.sin();               // [-0.2879, -0.9918, 0.9200, -0.5064]
double sum = tensor.sum();  // -0.86602...
//%%SNIPPET_END%% TensorOps
    }

    private static void tensorVertex() {
//%%SNIPPET_START%% TensorVertexCreate
GaussianVertex vertex = new GaussianVertex(new long[]{1, 100}, 0, 1);
//%%SNIPPET_END%% TensorVertexCreate
//%%SNIPPET_START%% TensorVertexInspection
DoubleTensor samples = vertex.sample();
samples.getShape();         //[1, 100]
samples.getLength();        //100
samples.getValue(0, 50);    //Returns the sample of the 50th Gaussian
//%%SNIPPET_END%% TensorVertexInspection
    }

    private static void tensorVector() {
//%%SNIPPET_START%% TensorVector
long[] shape = new long[]{3, 1};
DoubleVertex mu = new ConstantDoubleVertex(new double[]{1, 2, 3});
GaussianVertex vertex = new GaussianVertex(shape, mu, 0);
/** Creates a GaussianVertex that looks like...
* [ Gaussian(mu: 1, sigma: 0),
*   Gaussian(mu: 2, sigma: 0),
*   Gaussian(mu: 3, sigma: 0) ]
*/
//%%SNIPPET_END%% TensorVector
    }

    private static void tensorFinal() {
//%%SNIPPET_START%% TensorFinal
DoubleVertex muA = ConstantVertexFactory.of(new double[]{0.5, 1.5});
DoubleVertex A = new GaussianVertex(new long[]{1, 2}, muA, 1);
DoubleVertex B = ConstantVertexFactory.of(new double[]{3, 4});

DoubleVertex C = A.times(B);
DoubleVertex CObservation = new GaussianVertex(C, 1);
CObservation.observe(new double[]{6, 12});

//Use algorithm to find MAP or posterior samples for A and/or B
Optimizer optimizer = Optimizer.of(new BayesianNetwork(A.getConnectedGraph()));
optimizer.maxAPosteriori();

//Retrieve the most likely estimate using MAP estimation
DoubleTensor mostLikelyEstimate = A.getValue(); //approximately [2, 3]
//%%SNIPPET_END%% TensorFinal
    }
}
