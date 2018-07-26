package examples;

import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class CheatingStudentsTest {

    @Before
    public void setup() {
        KeanuRandom.setDefaultRandomSeed(1);
    }

    private final int numberOfStudents = 100;
    private final int numberOfYesAnswers = 35;

    @Test
    public void doesWorkWithHigherDimensionDescription() {
        double approximateProbabilityOfCheating = CheatingStudents.runWithFlips(numberOfStudents, numberOfYesAnswers);
        assertEquals(0.2, approximateProbabilityOfCheating, 0.05);
    }

    @Test
    public void doesWorkWithBinomial() {
        double approximateProbabilityOfCheating = CheatingStudents.runUsingBinomial(numberOfStudents, numberOfYesAnswers);
        assertEquals(0.2, approximateProbabilityOfCheating, 0.05);
    }

}
