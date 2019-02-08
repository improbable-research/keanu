package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.Statistics;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import lombok.Getter;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import javafx.application.Application;
import javafx.collections.ObservableList;
import javafx.geometry.HPos;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.layout.FlowPane;
import javafx.stage.Stage;

public class NUTSVis extends Application {


    @Override
    public void start(Stage primaryStage) {

        ModelForNUTSVis modelForVis = new RadonForVis();

        int sampleCount = modelForVis.getSampleCount();
        List<Vertex> toPlot = modelForVis.getToPlot();
        NUTS samplingAlgorithm = modelForVis.getSamplingAlgorithm();
        ProbabilisticModelWithGradient model = modelForVis.getModel();

        Map<Vertex, TracePlot> tracePlotByVertex = toPlot.stream()
            .collect(Collectors.toMap(
                v -> v,
                v -> {
                    String label = v.getLabel() == null ? "" : v.getLabel().toString();
                    return new TracePlot(label);
                }
            ));

        TracePlot tracePlotStepSize = new TracePlot("Step Size");

        List<LineChart<Number, Number>> lineCharts = tracePlotByVertex.values().stream()
            .map(v -> v.lineChart)
            .collect(Collectors.toList());

        lineCharts.add(tracePlotStepSize.getLineChart());

        FlowPane root = new FlowPane();
        root.setPrefWrapLength(1200);
        root.setColumnHalignment(HPos.LEFT);
        root.getChildren().addAll(lineCharts);

        Scene scene = new Scene(root);

        NetworkSamples posteriorSamples = samplingAlgorithm.getPosteriorSamples(
            model,
            toPlot,
            sampleCount
        );

//        networkSamplesGenerator.stream()
//            .limit(sampleCount)
//            .forEach(s -> {
//                tracePlotByVertex.forEach((v, plot) -> {
//                    plot.addPoint(s.get((Variable<DoubleTensor, ?>) v).scalar());
//                });
//            });

        tracePlotByVertex.forEach((v, plot) -> {
            plot.addPoints(
                posteriorSamples.get((Variable<DoubleTensor, ?>) v).asList().stream()
                    .map(t -> t.scalar())
                    .collect(Collectors.toList())
            );
        });

        Statistics statistics = samplingAlgorithm.getStatistics();

        List<Double> stepSizes = statistics.get(NUTS.Metrics.STEPSIZE);

        tracePlotStepSize.addPoints(stepSizes);

        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public static class TracePlot {

        @Getter
        final ObservableList<XYChart.Data<Number, Number>> data;

        @Getter
        final LineChart<Number, Number> lineChart;

        private AtomicInteger sampleNumber = new AtomicInteger(0);

        public TracePlot(String title) {
            //defining the axes
            final NumberAxis xAxis = new NumberAxis();
            xAxis.setAnimated(true);
            xAxis.setLabel("Sample Number");

            final NumberAxis yAxis = new NumberAxis();

            //creating the chart
            lineChart = new LineChart<>(xAxis, yAxis);
            lineChart.setTitle(title);

            lineChart.setAnimated(true);
            lineChart.setLegendVisible(false);

            //defining a series
            XYChart.Series<Number, Number> series = new XYChart.Series<>();
            lineChart.getData().add(series);

            lineChart.setCreateSymbols(false);

            data = series.getData();
        }

        public void addPoint(Number y) {
            data.add(new XYChart.Data<>(sampleNumber.getAndIncrement(), y));
        }

        public void addPoints(List<? extends Number> ys) {
            ys.forEach(this::addPoint);
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}
