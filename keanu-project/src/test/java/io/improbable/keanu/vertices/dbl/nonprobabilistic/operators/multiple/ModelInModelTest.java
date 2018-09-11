package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import java.io.*;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

public class ModelInModelTest {

    @Test
    public void canRunAModelInAModel() {
        DoubleVertex temperature = new ConstantDoubleVertex(25);

        Map<VertexLabel, DoubleVertex> inputs = new HashMap<>();
        inputs.put(new VertexLabel("Temperature"), temperature);

        String command = "./src/test/resources/model.sh {Temperature}";

        Function<Process, Map<VertexLabel, Double>> readInputs = new Function<Process, Map<VertexLabel, Double>>() {

            @Override
            public Map<VertexLabel, Double> apply(Process process) {
                Map<VertexLabel, Double> modelOutput = new HashMap<>();

                try (BufferedReader br = new BufferedReader(new FileReader("src/test/resources/rainOutput.txt"))) {
                    double chanceOfRainResult = Double.parseDouble(br.readLine());
                    modelOutput.put(new VertexLabel("ChanceOfRain"), chanceOfRainResult);
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                try (BufferedReader br = new BufferedReader(new FileReader("src/test/resources/humidityOutput.txt"))) {
                    double humidityResult = Double.parseDouble(br.readLine());
                    modelOutput.put(new VertexLabel("Humidity"), humidityResult);
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                return modelOutput;
            }

        };

        ModelVertex model = new ShellModelVertex(command, inputs, Collections.EMPTY_MAP, readInputs);
        DoubleVertex chanceOfRain = model.getModelOutputVertex(new VertexLabel("ChanceOfRain"));
        DoubleVertex humidity = model.getModelOutputVertex(new VertexLabel("Humidity"));

        DoubleVertex shouldIBringUmbrella = chanceOfRain.times(humidity);
        temperature.setAndCascade(10);
        System.out.println("Percentage chance of rain is: " + shouldIBringUmbrella.getValue().scalar());

        temperature.setAndCascade(20);
        System.out.println("Percentage chance of rain is: " + shouldIBringUmbrella.getValue().scalar());
    }

}
