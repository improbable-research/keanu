package com.example.coal;

import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.KeanuRandom;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class ModelTest {

    @Before
    public void setup() {
        KeanuRandom.setDefaultRandomSeed(1);
        VertexId.ID_GENERATOR.set(1);
    }

    @Test
    public void shouldFindCorrectSwitchYear() {

        //Load data from a csv file
        Data coalMiningDisasterData = Data.load("coal-mining-disaster-data.csv");

        //create my model using the data
        Model model = new Model(coalMiningDisasterData);
        model.run();

        int switchYear = model.results.getIntegerTensorSamples(model.switchpoint).getScalarMode();

        assertEquals(switchYear, 1890);
    }
}
