package io.improbable.docs;

import java.util.ArrayList;
import java.util.List;

public class LorenzModel {
    public static final double sigma = 10;
    public static final double beta = 2.66667;
    public static final double rho = 28;
    public static final double timeStep = 0.01;

    public List<Coordinates> runModel(int numTimeSteps) {

        Coordinates initialPosition = new Coordinates(2, 5, 4);

        List<Coordinates> results = new ArrayList<>();
        results.add(initialPosition);

        for (int i = 1; i <= numTimeSteps; i++) {
            Coordinates prevPosition = results.get(i - 1);
            results.add(getNextPosition(prevPosition));
        }

        return results;
    }

    private Coordinates getNextPosition(Coordinates currentPosition) {
        return new Coordinates(getNextX(currentPosition), getNextY(currentPosition), getNextZ(currentPosition));
    }

    private double getNextX(Coordinates currentPosition) {
        return currentPosition.x + timeStep * (sigma * (currentPosition.y - currentPosition.x));
    }

    private double getNextY(Coordinates currentPosition) {
        return currentPosition.y + timeStep * (currentPosition.x * (rho - currentPosition.z) - currentPosition.y);
    }

    private double getNextZ(Coordinates currentPosition) {
        return currentPosition.z + timeStep * (currentPosition.x * currentPosition.y - beta * currentPosition.z);
    }

    public static class Coordinates {
        public final double x;
        public final double y;
        public final double z;

        public Coordinates(double x, double y, double z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public String toString() {
            return "x:" + x + " y:" + y + " z:" + z;
        }
    }

}
