package rbf;

public class RunRBF
{

    // func for testing rbf and returns avg. percent error
    // rbfNeuralNetwork.getResults(double[] inputs); returns networks value
    // rbfNeuralNetwork.findCorrectAnswer(double[] inputs); returns actual value
    public static RBFNeuralNetwork testRBF(double[][] buildingSet, double[][] trainingSet, double[][] testingSet, double learningRate, int numbOfClusters)
    {
        RBFNeuralNetwork rbfNeuralNetwork = new RBFNeuralNetwork(buildingSet[0].length - 1, learningRate);
        rbfNeuralNetwork.createTrainingDataWithKClustering(numbOfClusters, buildingSet, buildingSet[0].length - 1);
        rbfNeuralNetwork.train(trainingSet);
        return rbfNeuralNetwork;
    }

    public static void main(String[] args)
    {
        RBFNeuralNetwork rbfNeuralNetwork = new RBFNeuralNetwork(3, 0.01);

        System.out.println("Hello world");
        rbfNeuralNetwork.createRandomTrainingDataWithKClustering(1000, 100000, 3);
        rbfNeuralNetwork.print();

        double x1 = Math.random() * 10 - 5;
        double x2 = Math.random() * 10 - 5;
        double x3 = Math.random() * 10 - 5;
        double[] inputs = {x1, x2, x3};
        x1 = Math.random() * 10 - 5;
        x2 = Math.random() * 10 - 5;
        x3 = Math.random() * 10 - 5;
        double[] inputs2 = {x1, x2, x3};
        x1 = Math.random() * 10 - 5;
        x2 = Math.random() * 10 - 5;
        x3 = Math.random() * 10 - 5;
        double[] inputs3 = {x1, x2, x3};
        x1 = Math.random() * 10 - 5;
        x2 = Math.random() * 10 - 5;
        x3 = Math.random() * 10 - 5;
        double[] inputs4 = {x1, x2, x3};
        x1 = Math.random() * 10 - 5;
        x2 = Math.random() * 10 - 5;
        x3 = Math.random() * 10 - 5;
        double[] inputs5 = {x1, x2, x3};

        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs));
        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs2));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs2));
        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs3));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs3));
        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs4));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs4));
        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs5));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs5));

        rbfNeuralNetwork.trainRandom(100000);

        System.out.println("After Training:");

        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs));
        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs2));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs2));
        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs3));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs3));
        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs4));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs4));
        System.out.println("Results: " + rbfNeuralNetwork.findCorrectAnswer(inputs5));
        System.out.println("rbf est." + rbfNeuralNetwork.getResult(inputs5));
    }
}
