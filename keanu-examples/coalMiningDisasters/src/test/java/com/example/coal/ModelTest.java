package com.example.coal;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class ModelTest {

    @Test
    public void shouldFindCorrectSwitchYear() {

        //Load data from a csv file
        Data coalMiningDisasterData = Data.load("coal-mining-disaster-data.csv");

        //create my model using the data
        Model model = new Model(coalMiningDisasterData);
        model.run();

        int switchYear = model.results.getIntegerTensorSamples(model.switchpoint).getScalarMode();

        assertTrue(switchYear == 1890);
    }
}
