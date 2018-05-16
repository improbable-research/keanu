package io.improbable.keanu.algorithms.tensormcmc;

public class MCMCVis {

    public static void main(String[] args){
        TensorHamiltonianTest test = new TensorHamiltonianTest();
        test.setup();
        test.samplesGaussian();
    }
}
