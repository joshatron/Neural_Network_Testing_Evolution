package nettest;

import feedforward.*;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;
import rbf.*;

/**
 * Created by joshua on 9/22/15.
 */

public class Trainer
{
    static final Random rand = new Random();
    
    /***********************Test Values*****************************
     * 
     *                      * = default
     *
     * Default values are set if the value's respective array is either null or
     * empty (length = 0). 
     * 
     *  Sizes: 1000, 50000, 100000*, 500000, 1000000
     *
     *  Repeats:
     *  {100,5000}, {50000,100}, {5000000,1}, {1,1}*
     *
     *  Dimensions:
     *  2, 3, 4*, 5, 6
     *
     *  RBF basis function: gaussian*, cubed, multi-quad
     *
     *  FF activation function: 
     *  linear, log*
     *
     *  Learning rate:
     *  0.1, 0.05, 0.01*, 0.005, 0.001
     *
     *  Clusters {hidden num, number of clusters}:
     *  {500,50}, {10000,50}, {100000,50} {500,100}, {10000,100}*, {100000,100}, {500,500}, {10000,500}, {100000, 500}
     *
     *  momentums
     *  .1, .01*, .001
     *
     *  hidden (hidden size)
     *  0, 1*, 2
     *
     * 
     *************************************************************
     */
    public static void main(String[] args) throws IOException {
        
        // Dataset sizes
        int[] sizes = new int[] 
        {
                1000,50000,100000,500000,1000000,
        };
        
        // Numbers of inputs to the Rosenbrock function
        int[] dimensions = new int[] 
        {
            4
        };
        
        // Numbers of times to repeat training on the same data
        int[][] repeats = new int[][] 
        {
            {1,1}
        };
        
        // Basis functions for the RBF neural network
        int[] rbfBasisFunction = new int[] 
        {
            0
        };
        
        // For the Feed Forward net
        ActivationFunction[] activationFunction = new ActivationFunction[] 
        {
            ActivationFunction.LOGISTIC
        };
        
        // Learning rates
        double[] learningRate = new double[]
        {
            0.01
        };
        
        // { Numbers of nodes in each hidden layer, number of clusters }
        int[][] clusters = new int[][]
        {
                {10000,100}
        };
        
        // Momentums for the Feed Forward neural net
        double[] momentum = new double[]
        {
            0.01
        };
        
        // Numbers of hidden layers.
        int[] hiddenNum = new int[]
        {
            1
        };
        
        System.out.println("Would you like to specify parameters? (y/n)");
        Scanner in = new Scanner(System.in);
        String input = in.nextLine().toLowerCase();
        if (input.charAt(0) == 'y') {
            setExperiment();
        } else {
            runExperiment(sizes,dimensions,repeats,rbfBasisFunction,activationFunction,learningRate,clusters,momentum,hiddenNum);
        }
    }
    
    /**
     * Enable the user to input the parameters of one experiment.
     */
    public static void setExperiment() {
        Scanner in = new Scanner(System.in);
        System.out.println("-----Set the parameters for the feed forward neural net-----");
        System.out.print("Data set size: ");
        int size = in.nextInt();
        System.out.println();
        System.out.print("Number of inputs: ");
        int[] sizesLayers = new int[3];
        sizesLayers[0] = in.nextInt();
        System.out.println();
        System.out.print("Number of Hidden Layers: ");
        sizesLayers[1] = in.nextInt();
        System.out.println();
        System.out.print("Size of Hidden Layers: ");
        sizesLayers[2] = in.nextInt();
        System.out.println();
        System.out.print("Learning rate: ");
        double learningRate = in.nextDouble();
        System.out.println();
        System.out.print("Momentum: ");
        double momentum = in.nextDouble();
        System.out.println();
        System.out.print("Activation function ('linear' or 'logistic'): ");
        String selection = in.nextLine().toLowerCase();
        System.out.println();
        ActivationFunction activationFunction;
        if (selection.charAt(1) == 'o' || selection.charAt(2) == 'g') {
            activationFunction = ActivationFunction.LOGISTIC;
            selection = "Logistic";
        } else {
            activationFunction = ActivationFunction.LINEAR;
            selection = "Linear";
        }
        System.out.print("Number of clusters: ");
        int numbOfClusters = in.nextInt();
        System.out.println();
        System.out.print("RBF basis function (0 = Gaussian, 1 = cubed, 2 = multi-quad): ");
        int rbfBasisFunction = in.nextInt();
        System.out.println();
        
        runExperiment(new int[]{size},
                      new int[]{sizesLayers[0]},
                      new int[][]{{1,1}},
                      new int[]{rbfBasisFunction},
                      new ActivationFunction[]{activationFunction},
                      new double[]{learningRate},
                      new int[][]{{sizesLayers[2],numbOfClusters}},
                      new double[]{momentum},new int[]{sizesLayers[1]});
    }
    
    /**
     * Test a Feedforward neural net using 5x2-fold cross validation.
     * @param size
     * @param dimensions
     * @param repeats
     * @param rbfBasisFunction
     * @param activationFunction
     * @param learningRate
     * @param clusters
     * @param momentums
     * @param hidden
     * @param dataset
     * @return 2D array of results
     */
    public static double[][] trainFF(
                int size, 
                int dimensions, 
                int[] repeats, 
                int rbfBasisFunction, 
                ActivationFunction activationFunction,
                double learningRate,
                int[] clusters,
                double momentums,
                int hidden,
                double[][] dataset) 
    {
        // Create the sizes array used by the feed forward neural net
        int[] sizesArray = new int[hidden + 2];
        sizesArray[0] = dimensions + 1;
        for(int i = 1; i < sizesArray.length - 1; i++) {
            sizesArray[i] = clusters[0];
        }
        sizesArray[sizesArray.length-1] = 2;
        
        // Initialize the storage for the output of this test.
        double[][] output = new double[7][10];  // Bundle of statistics
        double[] confidences = new double[size*10]; // Store all confidence values
        double confidenceAR = 0.0;  // Sum of confidences above when guessed right
        double confidenceAW = 0.0;  // Sum of confidences above when guessed wrong
        double confidenceBR = 0.0;  // Sum of confidences below when guessed wrong
        double confidenceBW = 0.0;  // Sum of confidences below when guessed right
        System.out.println("Feed Forward Neural Network:");
        // For 5 repetitions of 2-fold cross validation
        for (int i = 0; i < 5; i++) {
            int count = i + 1;
            System.out.println("Two-fold cross validation repetition " + count);
            double[][][] datasets = partitionData(dataset);
            
            // First: train on datasets[0], test on datasets[1]
            FeedForwardNeuralNetwork ff = RunFeedForward.run(datasets[0], hidden, sizesArray, learningRate, momentums, activationFunction, repeats[1]);
            
            double confidenceSum = 0.0;
            double aboveRight = 0.0;
            double aboveWrong = 0.0;
            double belowRight = 0.0;
            double belowWrong = 0.0;
            
            for(int j = 0; j < datasets[1].length; j++) {
                // Index of the output of the Rosenbrock function
                int index = datasets[1][j].length - 1;
                // Actual output of the Rosenbrock function
                double actualValue = datasets[1][j][index];
                // Value plus or minus a constant percent of the actual value
                double offsetValue = plusOrMinus10(actualValue);
                
                // Compute the confidences that the offset value is above or below the function
                datasets[1][j][index] = offsetValue;
                double[] prediction = ff.compute(datasets[1][j]);
                datasets[1][j][index] = actualValue;

                boolean abovePredicted =  prediction[1] < prediction[0];
                boolean aboveActual = actualValue < offsetValue;
                double confidence;
                if (abovePredicted && aboveActual) {
                    confidence = prediction[0];
                    confidenceAR += confidence;
                    aboveRight += 1.0;
                } else if (abovePredicted) {
                    confidence = prediction[0];
                    confidenceAW += confidence;
                    aboveWrong += 1.0;
                } else if (aboveActual) {
                    confidence = prediction[1];
                    confidenceBW += confidence;
                    belowWrong += 1.0;
                } else {
                    confidence = prediction[1];
                    confidenceBR += confidence;
                    belowRight += 1.0;
                }
                confidences[size*i+j*2] = confidence;
                confidenceSum += confidence;
            }
        
            confidenceSum /= datasets[1].length;
            
            output[i][0] = aboveRight;
            output[i][1] = aboveWrong;
            output[i][2] = belowRight;
            output[i][3] = belowWrong;
            output[i][4] = confidenceSum;
            
            
            // Second: train on datasets[1], test on datasets[0]
            ff = RunFeedForward.run(datasets[1], hidden, sizesArray, learningRate, momentums, activationFunction, repeats[1]);
        
            confidenceSum = 0.0;
            aboveRight = 0.0;
            aboveWrong = 0.0;
            belowRight = 0.0;
            belowWrong = 0.0;
            
            for(int j = 0; j < datasets[0].length; j++) {
                // Index of the output of the Rosenbrock function
                int index = datasets[0][j].length - 1;
                // Actual output of the Rosenbrock function
                double actualValue = datasets[0][j][index];
                // Value plus or minus a constant percent of the actual value
                double offsetValue = plusOrMinus10(actualValue);
                
                // Compute the confidences that the offset value is above or below the function
                datasets[0][j][index] = offsetValue;
                double[] prediction = ff.compute(datasets[0][j]);
                datasets[0][j][index] = actualValue;
                
                // True iff the prediction is above the function
                boolean abovePredicted =  prediction[1] < prediction[0];
                
                // True iff the offset value is above the function 
                boolean aboveActual = actualValue < offsetValue;
                
                double confidence;  // This will be the selected confidence
                if (abovePredicted && aboveActual) {
                    confidence = prediction[0];
                    confidenceAR += confidence;
                    aboveRight += 1.0;
                } else if (abovePredicted) {
                    confidence = prediction[0];
                    confidenceAW += confidence;
                    aboveWrong += 1.0;
                } else if (aboveActual) {
                    confidence = prediction[1];
                    confidenceBW += confidence;
                    belowWrong += 1.0;
                } else {
                    confidence = prediction[1];
                    confidenceBR += confidence;
                    belowRight += 1.0;
                }
                
                // Store confidence data
                confidences[size*i+j*2+1] = confidence;
                confidenceSum += confidence;
            }
        
            confidenceSum /= datasets[0].length;
            
            // Store data from every train/test separately
            output[i][5] = aboveRight;
            output[i][6] = aboveWrong;
            output[i][7] = belowRight;
            output[i][8] = belowWrong;
            output[i][9] = confidenceSum;
        }
        
        // Store all confidence values
        output[5] = confidences;
        output[6] = new double[]{confidenceAR,confidenceAW,confidenceBW,confidenceBR};
        return output;
    }
    
    /**
     * Train and test the RBF net using 5x2-fold cross validation.
     * @param size
     * @param dimensions
     * @param repeats
     * @param rbfBasisFunction
     * @param activationFunction
     * @param learningRate
     * @param clusters
     * @param momentums
     * @param hidden
     * @param dataset
     * @return 2D array of results
     */
    public static double[][] trainRBF(
                int size, 
                int dimensions, 
                int[] repeats, 
                int rbfBasisFunction, 
                ActivationFunction activationFunction,
                double learningRate,
                int[] clusters,
                double momentums,
                int hidden,
                double[][] dataset)
        {
        
        double[][] output = new double[4][10];
        double[] errors = new double[size*10];
        double[] allValues = new double[size*10];
        double[] allGuesses = new double[size*10];
        
        double aboveRight = 0.0;    // Total above and correct guesses
        double aboveWrong = 0.0;    // Total above and incorrect guesses
        double belowRight = 0.0;    // Total below and incorrect guesses
        double belowWrong = 0.0;    // Total below and correct guesses
        double varianceSum = 0.0;   // Sum of the error
        double realValue = 0.0;     // sum of the actual values
        double estimatedValue = 0.0;    // Sum of the approximated values
        double minError = 99999999.0;   // minimum error
        double maxError = 0.0;          // maximum error
        double percentError = 0.0;      // sum of the percent errors
        System.out.println("RBF Neural Network:");
        // For 5 repetitions of 2-fold cross validation
        for (int i = 0; i < 5; i++) {
            System.out.println("Running cross-validation, on part: " + i);
            int count = i + 1;
            System.out.println("Two-fold cross validation repetition " + count);
            double[][][] datasets = partitionData(dataset);
            double[][] buildingData;

            int buildingSize = Math.min(clusters[0], datasets[0].length);
            buildingData = new double[buildingSize][];

            for (int j = 0; j < buildingSize; j++)
            {
                buildingData[j] = datasets[0][j];
            }

            // First: train on datasets[0], test on datasets[1]
            long startTime = System.currentTimeMillis();
            RBFNeuralNetwork rbf = RunRBF.testRBF(buildingData, datasets[0], learningRate, clusters[1], rbfBasisFunction, repeats[1]);
            long end = System.currentTimeMillis();
            System.out.println("Run time: " + (end - startTime));

            for(int j = 0; j < datasets[1].length; j++) {
                // Index of the output of the Rosenbrock function
                int index = datasets[1][j].length - 1;
                // Actual output of the Rosenbrock function
                double actualValue = datasets[1][j][index];
                realValue += actualValue;
                // Value plus or minus a constant percent of the actual value
                double offsetValue = plusOrMinus10(actualValue);
                // Approximated value
                double predictedValue = rbf.getResult(datasets[1][j]);
                estimatedValue += predictedValue;
                
                // Variance
                double variance = Math.abs(predictedValue - actualValue);
                
                // Store statistics aquired thus far
                errors[size*i+j*2] = variance;
                allValues[size*i+j*2] = realValue;
                allGuesses[size*i+j*2] = predictedValue;
                
                // Calculate percent error
                percentError += 100* ((predictedValue - actualValue) / actualValue);
                
                varianceSum += variance;
                if (minError > variance) {
                    minError = variance;
                }
                
                if (maxError < variance) {
                    maxError = variance;
                }
                
                // Predict above or below the function
                boolean abovePredicted = rbf.aboveValue(datasets[1][j], offsetValue);

                boolean aboveActual = actualValue < offsetValue;

                if (abovePredicted && aboveActual) {
                    aboveWrong += 1.0;
                } else if (abovePredicted) {
                    aboveRight += 1.0;
                } else if (aboveActual) {
                    belowRight += 1.0;
                } else {
                    belowWrong += 1.0;
                }
            }
            
            // Cross validation, swap testing and training sets
            rbf = RunRBF.testRBF(datasets[1], datasets[1], learningRate, clusters[1], rbfBasisFunction, repeats[1]);
        
            for(int j = 0; j < datasets[0].length; j++) {
                // Index of the output of the Rosenbrock function
                int index = datasets[0][j].length - 1;
                // Actual output of the Rosenbrock function
                double actualValue = datasets[0][j][index];
                realValue += actualValue;
                // Value plus or minus a constant percent of the actual value
                double offsetValue = plusOrMinus10(actualValue);
                // Approximated value
                double predictedValue = rbf.getResult(datasets[0][j]);
                estimatedValue += predictedValue;
                
                
                double variance = Math.abs(predictedValue - actualValue);
                // Store statistics aquired thus far
                errors[size*i+j*2+1] = variance;
                allValues[size*i+j*2+1] = realValue;
                allGuesses[size*i+j*2+1] = predictedValue;
                percentError += 100* ((predictedValue - actualValue) / actualValue);
                
                varianceSum += variance;
                if (minError > variance) {
                    minError = variance;
                }
                
                if (maxError < variance) {
                    maxError = variance;
                }
                
                // Predict above or below the function
                boolean abovePredicted = rbf.aboveValue(datasets[0][j], offsetValue);

                boolean aboveActual = actualValue < offsetValue;
                
                if (abovePredicted && aboveActual) {
                    aboveWrong += 1.0;
                } else if (abovePredicted) {
                    aboveRight += 1.0;
                } else if (aboveActual) {
                    belowRight += 1.0;
                } else {
                    belowWrong += 1.0;
                }
            }
        }
        
        // Store all data in the output array
        output[0][0] = aboveRight;
        output[0][1] = aboveWrong;
        output[0][2] = belowRight;
        output[0][3] = belowWrong;
        output[0][4] = varianceSum;
        output[0][5] = realValue;
        output[0][6] = estimatedValue;
        output[0][7] = minError;
        output[0][8] = maxError;
        output[0][9] = percentError;
        output[1] = errors;
        output[2] = allValues;
        output[3] = allGuesses;
        
        return output;
    }
    
    /**
     * Randomly add to or subtract from the input a fixed percentage of the input
     * @param output output of the Rosenbrock function
     * @return output plus or minus a fixed percentage.
     */
    public static double plusOrMinus10(double output) {
        double newVal = output + output * 1 * Math.pow(-1, (double)(rand.nextInt(2) + 1)); //Math.pow(-1, (double)(rand.nextInt(2) + 1)) * 2 * output * Math.random() + output;
        return newVal;
    }
    
    /**
     * Swap training and testing sets
     * @param trainingAndTesting
     * @return same data, but first index is swapped
     */
    private static double[][][] swapTrainingAndTesting(double[][][] trainingAndTesting) {
        for (int i = 0; i < trainingAndTesting[0].length; i++) {
            int lastIndex = trainingAndTesting[0][i].length - 1;
            int secondLast = lastIndex - 1;
            
            double swap = trainingAndTesting[0][i][lastIndex];
            trainingAndTesting[0][i][lastIndex] = trainingAndTesting[0][i][secondLast];
            trainingAndTesting[0][i][secondLast] = swap;
            
            swap = trainingAndTesting[1][i][lastIndex];
            trainingAndTesting[1][i][lastIndex] = trainingAndTesting[1][i][secondLast];
            trainingAndTesting[1][i][secondLast] = swap;
        }
        
        return trainingAndTesting;
    }
    
    /**
     * Randomly partition the dataset into two equal subsets, one training and
     * one testing
     * @param dataset data to be partitioned
     * @return two random halves of the data set
     */
    private static double[][][] partitionData(double[][] dataset) {
        
        int numInputs = dataset[0].length;
        int size = dataset.length;
        
        // Two halves of the dataset
        double[][] trainingData = new double[size/2][numInputs];
        double[][] testingData = new double[size/2][numInputs];
        
        int trainingIndex = 0;
        int testingIndex = 0;
        int dataIndex = 0;
        int half = size/2;
        
        while(trainingIndex < half && testingIndex < half) {
            // Flip a coin...
            if (rand.nextBoolean()) {
                // If heads, add the next instance of the dataset to the training set
                for (int i = 0; i < numInputs; i++) {
                    trainingData[trainingIndex][i] = dataset[dataIndex][i];
                }
                
                trainingIndex++;
                dataIndex++;
            } else {
                // If tails, add the next instance of the dataset to the testing set
                for (int i = 0; i < numInputs; i++) {
                    testingData[testingIndex][i] = dataset[dataIndex][i];
                }
                
                testingIndex++;
                dataIndex++;
            }
        }
        
        // If the training index is still not full, give it the remaining data
        while(trainingIndex < half) {
            for (int i = 0; i < numInputs; i++) {
                trainingData[trainingIndex][i] = dataset[dataIndex][i];
            }
                
            trainingIndex++;
            dataIndex++;
        }
        
        // If the testing index is still not full, give it the remaining data
        while(testingIndex < half) {
            for (int i = 0; i < numInputs; i++) {
                testingData[testingIndex][i] = dataset[dataIndex][i];
            }
            
            testingIndex++;
            dataIndex++;
        }
        
        return new double[][][] { trainingData, testingData };
    }
    
    /**
     * Run an experiment, iterating through all possible combinations of parameters.
     * @param sizes
     * @param dimensions
     * @param repeats
     * @param rbfBasisFunction
     * @param activationFunction
     * @param learningRate
     * @param clusters {number of hidden nodes in each layer, number of clusters}
     * @param momentum
     * @param hiddenNum number of hidden layers
     */
    public static void runExperiment(
            int[] sizes, 
            int[] dimensions, 
            int[][] repeats, 
            int[] rbfBasisFunction, 
            ActivationFunction[] activationFunction, 
            double[] learningRate, 
            int[][] clusters, 
            double[] momentum, 
            int[] hiddenNum) {
        
        // Check for empty and null-valued arguments and initialize to default values.
        if (sizes == null || sizes.length == 0) {
            sizes = new int[]{100000};
        }
        if (dimensions == null || dimensions.length == 0) {
            dimensions = new int[]{4};
        }
        if (repeats == null || repeats.length == 0) {
            repeats = new int[][]{{1,1}};
        }
        if (rbfBasisFunction == null || rbfBasisFunction.length == 0) {
            rbfBasisFunction = new int[]{0};
        }
        if (activationFunction == null || activationFunction.length == 0) {
            activationFunction = new ActivationFunction[]{ActivationFunction.LOGISTIC};
        }
        if (learningRate == null || learningRate.length == 0) {
             learningRate = new double[]{0.01};
        }
        if (clusters == null || clusters.length == 0) {
            clusters = new int[][]{{100,10000}};
        }
        if (momentum == null || momentum.length == 0) {
            momentum = new double[]{0.01};
        }
        if (hiddenNum == null || hiddenNum.length == 0) {
            hiddenNum = new int[]{1};
        }
        
        // calculate the number of experiments
        int experimentCount = 
                sizes.length * 
                repeats.length * 
                dimensions.length * 
                rbfBasisFunction.length * 
                activationFunction.length * 
                learningRate.length * 
                clusters.length * 
                momentum.length * 
                hiddenNum.length;
        
        
        double[][][][] results = new double[2][experimentCount][][];
        int experimentIndex = 0;
        
        // Loop through all possible combinations of variables (most arrays should be of size 1)
        for (int a = 0; a < sizes.length; a++) {
            for (int b = 0; b < dimensions.length; b++) {
                // Retrieve a dataset of size a with b inputs, or generate one if it does not exist
                double[][] datasets = DataTools.getDataFromFile(dimensions[b], sizes[a]);
                
                for (int c = 0; c < repeats.length; c++) {
                    for (int d = 0; d < rbfBasisFunction.length; d++) {
                        for (int e = 0; e < activationFunction.length; e++) {
                            for (int f = 0; f < learningRate.length; f++) {
                                for (int g = 0; g < clusters.length; g++) {
                                    for (int h = 0; h < momentum.length; h++) {
                                        for (int i = 0; i < hiddenNum.length; i++) {
                                            
                                            // Print information about the next experiment
                                            System.out.println("Experiment " + experimentIndex);
                                            System.out.println("Number of training examples: " + sizes[a]);
                                            System.out.println("Number of inputs: " + dimensions[b]);
                                            System.out.println("Repeats: {" + repeats[c][0] + "," + repeats[c][1] + "}");
                                            System.out.println("RBF basis function: " + rbfBasisFunction[d]);
                                            System.out.println("FF activation function: " + activationFunction[e]);
                                            System.out.println("Learning rate: " + learningRate[f]);
                                            System.out.println("Clusters: " + clusters[g][1]);
                                            System.out.println("Momentum: " + momentum[h]);
                                            System.out.println("Hidden layers: " + hiddenNum[i]);
                                            System.out.println();
                                            
                                            
                                            // Train and test an RBF neural net
                                            results[0][experimentIndex] = trainRBF(
                                                    sizes[a], 
                                                    dimensions[b], 
                                                    repeats[c], 
                                                    rbfBasisFunction[d], 
                                                    activationFunction[e], 
                                                    learningRate[f], 
                                                    clusters[g],
                                                    momentum[h], 
                                                    hiddenNum[i], 
                                                    datasets);
                                            
                                            double aboveRight = results[0][experimentIndex][0][0];
                                            double aboveWrong = results[0][experimentIndex][0][1];
                                            double belowRight = results[0][experimentIndex][0][2];
                                            double belowWrong = results[0][experimentIndex][0][3];
                                            double varianceSum = results[0][experimentIndex][0][4];
                                            double realSum = results[0][experimentIndex][0][5];
                                            double predictionSum = results[0][experimentIndex][0][6];
                                            double minError = results[0][experimentIndex][0][7];
                                            double maxError = results[0][experimentIndex][0][8];
                                            double percentError =  results[0][experimentIndex][0][9] / (sizes[a] * 10);
                                            
                                            double averageError = varianceSum / (sizes[a] * 10);
                                            
                                            double standardDeviation = 0.0;
                                            double averageReal = realSum  / (sizes[a] * 10);
                                            double averagePrediction = predictionSum / (sizes[a] * 10);
                                            
                                            double[] errors = results[0][experimentIndex][1];
                                            for (int j = 0; j < errors.length; j++) {
                                                double error = results[0][experimentIndex][1][j];
                                                standardDeviation += Math.pow(error-averageError, 2.0);
                                            }

                                            standardDeviation /= 10 * sizes[a] - 1;
                                            standardDeviation = Math.sqrt(standardDeviation);
                                            
                                            System.out.println("------------RBF Neural Network------------");
                                            System.out.println("    Correct above guesses: " + aboveRight);
                                            System.out.println("  Incorrect above guesses: " + aboveWrong);
                                            System.out.println("    Correct below guesses: " + belowRight);
                                            System.out.println("  Incorrect below guesses: " + belowWrong);
                                            System.out.println("       Average real value: " + averageReal);
                                            System.out.println("  Average predicted value: " + averagePrediction);
                                            System.out.println("            Average error: " + averageError);
                                            System.out.println("    Average percent error: " + percentError);
                                            System.out.println("  Standard dev. of errors: " + standardDeviation);
                                            System.out.println("            Minimum error: " + minError);
                                            System.out.println("            Maximum error: " + maxError);
                                            System.out.println("Percent correctly guessed: " + ((aboveRight + belowRight) / (aboveRight + belowRight + aboveWrong + belowWrong) * 100));
                                            System.out.println();
                                            
                                            
                                            // Train and test a Feedforward neural net
                                            results[1][experimentIndex] =  trainFF(
                                                    sizes[a],
                                                    dimensions[b],
                                                    repeats[c],
                                                    rbfBasisFunction[d],
                                                    activationFunction[e],
                                                    learningRate[f],
                                                    clusters[g],
                                                    momentum[h],
                                                    hiddenNum[i],
                                                    datasets);

                                            double[] statistics = new double[5];
                                            for (int j = 0; j < 5; j++) {
                                                for (int k = 0; k < 10; k++) {
                                                    statistics[k%5] += results[1][experimentIndex][j][k];
                                                }
                                            }

                                            statistics[4] /= 10;

                                            double[] confidences = results[1][experimentIndex][6];

                                            confidences[0] /= statistics[0];
                                            confidences[1] /= statistics[1];
                                            confidences[2] /= statistics[2];
                                            confidences[3] /= statistics[3];


                                            System.out.println("--------Feed Forward Neural Network--------");
                                            System.out.println("  Correct above guesses: " + statistics[0]);
                                            System.out.println("Incorrect above guesses: " + statistics[1]);
                                            System.out.println("  Correct below guesses: " + statistics[2]);
                                            System.out.println("Incorrect below guesses: " + statistics[3]);
                                            System.out.println("     Average confidence: " + statistics[4]);
                                            System.out.println("Average confidence when...");
                                            System.out.println("      Correct and above: " + confidences[0]);
                                            System.out.println("    Incorrect and above: " + confidences[1]);
                                            System.out.println("      Correct and below: " + confidences[2]);
                                            System.out.println("    Incorrect and below: " + confidences[3]);
                                            System.out.println();
                                            
                                            experimentIndex++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
