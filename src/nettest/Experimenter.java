package nettest;

import feedforward.*;
import training.*;

import org.json.JSONObject;

/**
 * 
 * @author davej
 */
public class Experimenter {
    // JSONObject holding data for the initial neural network.
    private final JSONObject initialNet;
    
    // Indices of arrays wrapped by the Results class.
    private static final int trueIndex = Results.Index.ACTUAL.ordinal();
    private static final int predictedIndex = Results.Index.PREDICTED.ordinal();
    private static final int confidenceIndex = Results.Index.CONFIDENCE.ordinal();
    
    // Trainer objects that will be used in experiments.
    private final Trainer[] trainers;
    
    /**
     * Constructor passes an array of Trainer objects to iterate through this
     * experiment. 
     * @param in_initialNet JSONObject holding parameters for the initial neural net.
     * @param in_trainers   Array of Trainer objects.
     */
    public Experimenter(JSONObject in_initialNet, Trainer[] in_trainers) {
        initialNet = in_initialNet;
        trainers = in_trainers;
    }
    
    /**
     * Run every Trainer on the dataset.
     * @param dataset  Array of different datasets to test.
     */
    public void run(double[][] dataset) {
        for (Trainer trainer : trainers) {
            long startTime = System.currentTimeMillis();
            Results[][] results = test(trainer,dataset);
            double[] percentsCorrect = new double[5];
            double[] averageConfidences = new double[5];
            
            double percentCorrect = 0.0;
            double averageConfidence = 0.0;
            
            for (int i = 0; i < 5; i++) {
                percentsCorrect[i] = ( results[i][0].percentCorrect() +
                                       results[i][1].percentCorrect() ) / 2;

                System.out.println("Percent Correct: " + results[i][0].percentCorrect() * 100);
                System.out.println("Percent Correct: " + results[i][1].percentCorrect() * 100);
                
                percentCorrect += percentsCorrect[i];
                
                averageConfidences[i] = ( results[i][0].getMean(Results.Index.CONFIDENCE) +
                                          results[i][1].getMean(Results.Index.CONFIDENCE) ) / 2;
                
                averageConfidence += averageConfidences[i];
            }
            
            percentCorrect /= 5.0;
            averageConfidence /= 5.0;

            long end = System.currentTimeMillis();
            
            System.out.println("Trainer: " + trainer.toString());
            System.out.println("Average Percent correct: " + (percentCorrect * 100));
            System.out.println("Mean confidence: " + averageConfidence);
            System.out.println("Average run time: " + (end - startTime) / 10 + "ms");
        }
    }
    
    /**
     * Run 5x2-fold cross validation on one dataset and one trainer.
     * Returns one pair of Results objects for every iteration of cross
     * validation, for a total of 5 pairs.
     * @param trainer   Trainer used to train the neural network
     * @param datasets  Training and testing datasets for cross validation
     * @return Results[5][2], 5 pairs of Results objects.
     */
    public Results[][] test(Trainer trainer, double[][] dataset) {
        Results[][] results = new Results[5][];
        for (int i = 0; i < 5; i++) {
            double[][][] datasets = DataTools.partitionData(dataset);
            results[i] = crossValidate(trainer, datasets[0], datasets[1]);
        }
        
        return results;
    }
    
    /**
     * Runs one iteration of cross validation, returning a pair of results objects.
     * @param trainer   Trainer used to train the neural network.
     * @param datasetA  Training/testing dataset
     * @param datasetB  Testing/training dataset
     * @return Results[2], one pair of results objects.
     */
    private Results[] crossValidate(Trainer trainer, double[][] datasetA, double[][] datasetB) {
        Results[] results = new Results[2];

        FeedForwardNeuralNetwork neuralNet = trainer.run(initialNet(), datasetB);

        double[][] data = new double[5][];
        
        data[trueIndex] = new double[datasetA.length];
        data[predictedIndex] = new double[datasetA.length];
        data[confidenceIndex] = new double[datasetA.length];
        
        results[0] = new Results(data);
        
        for (int i = 0; i < datasetA.length; i++) {
            int classIndex = datasetA[i].length - 1;
            double trueValue = datasetA[i][classIndex];
            double[] inputs = new double[datasetA[i].length - 1];
            for(int j = 0; j < inputs.length; j++)
            {
                inputs[j] = datasetA[i][j];
            }
            double[] confidences = neuralNet.compute(inputs);
            
            double predictedValue, confidence;
            
            confidence = -1;
            predictedValue = -1;
            
            for (int j = 0; j < confidences.length; j++) {
                if (confidences[j] > confidence) {
                    predictedValue = j;
                    confidence = confidences[j];
                }
            }
            
            data[predictedIndex][i] = predictedValue;
            data[trueIndex][i] = trueValue;
            data[confidenceIndex][i] = confidence;
        }
        
        neuralNet = trainer.run(initialNet(), datasetB);
        
        data = new double[5][];
        
        data[trueIndex] = new double[datasetB.length];
        data[predictedIndex] = new double[datasetB.length];
        data[confidenceIndex] = new double[datasetB.length];
        
        results[1] = new Results(data);
        
        for (int i = 0; i < datasetB.length; i++) {
            int classIndex = datasetB[i].length - 1;
            double trueValue = datasetB[i][classIndex];
            double[] inputs = new double[datasetB[i].length - 1];
            for(int j = 0; j < inputs.length; j++)
            {
                inputs[j] = datasetB[i][j];
            }
            double[] confidences = neuralNet.compute(inputs);

//            for(int j = 0; j < confidences.length; j++) {
//                System.out.format("%.5f,",confidences[j]);
//            }
            System.out.println(confidences.length);
            
            double predictedValue, confidence;
            
            confidence = -1;
            predictedValue = -1;
            
            for (int j = 0; j < confidences.length; j++) {
                if (confidences[j] > confidence) {
                    predictedValue = j;
                    confidence = confidences[j];
                }
            }
            
            data[predictedIndex][i] = predictedValue;
            data[trueIndex][i] = trueValue;
            data[confidenceIndex][i] = confidence;
        }
        
        return results;
    }
    
    /**
     * Retrieve the initial neural network.
     * @return FeedForwardNeuralNetwork
     */
    private FeedForwardNeuralNetwork initialNet() {
        return new FeedForwardNeuralNetwork(initialNet);
    }
}
