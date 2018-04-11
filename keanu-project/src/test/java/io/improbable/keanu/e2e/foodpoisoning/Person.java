package io.improbable.keanu.e2e.foodpoisoning;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpLambda;

public class Person {
    Flip didEatOysters = new Flip(0.4, FoodPoisoningTest.rand);
    Flip didEatLamb = new Flip(0.4, FoodPoisoningTest.rand);
    Flip didEatPoo = new Flip(0.4, FoodPoisoningTest.rand);
    Flip isIll;

    public Person(Vertex<Boolean> oystersInfected,
                  Vertex<Boolean> lambInfected,
                  Vertex<Boolean> toiletInfected) {

        BoolVertex ingestedPathogen =
                didEatOysters.and(oystersInfected).or(
                        didEatLamb.and(lambInfected).or(
                                didEatPoo.and(toiletInfected)
                        )
                );

        DoubleUnaryOpLambda<Boolean> pIll = new DoubleUnaryOpLambda<>(ingestedPathogen, (i) -> i ? 0.9 : 0.01);
        isIll = new Flip(pIll, FoodPoisoningTest.rand);
    }
}
