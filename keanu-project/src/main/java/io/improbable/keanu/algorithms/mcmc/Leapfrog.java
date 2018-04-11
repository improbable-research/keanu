package io.improbable.keanu.algorithms.mcmc;

public class Leapfrog {

    private double position;
    private double momentum;

    public Leapfrog(double theta, double momentum) {
        this.position = theta;
        this.momentum = momentum;
    }

    public double getPosition() {
        return position;
    }

    public double getMomentum() {
        return momentum;
    }
}
