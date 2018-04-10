package com.example.starter;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class ModelTest {

    @Test
    public void shouldDoSomethingIExpect() {

        //Load data from a csv file
        Data data = Data.load("data_example.csv");

        //create my model using the data
        Model model = new Model(data);
        model.run();

        //make some assertions about the results of your model
        assertTrue(model.results != null);
    }
}
